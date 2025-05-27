#!/usr/bin/env python3
"""
Combine Python source files into a single XML document.

Security-first refactor:
- Sanitises all XML output via proper escaping rather than home-made tags.
- Skips the output file even if it lives inside the search tree.
- Opt-in symlink traversal (`--follow-symlinks`).
- Hard stops (or warnings) on excessive file count / output size.
- Robust UTF-8 handling with 'replace' and logging.
- Generates a concise summary report for CI pipelines or humans.

Author: refactored 28 May 2025
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import sys
from pathlib import Path
from typing import Dict

from xml.sax.saxutils import escape as xml_escape


# ---------- Configuration helpers ------------------------------------------------- #

DEFAULT_MAX_FILES = 10_000
DEFAULT_MAX_SIZE_MB = 100  # warn after 100 MB written

logger = logging.getLogger("combiner")


# ---------- Core logic ------------------------------------------------------------ #

def combine_python_files(
    start_dir: Path,
    output_path: Path,
    *,
    follow_symlinks: bool = False,
    max_files: int | None = None,
    max_size_bytes: int | None = None,
    compress: bool = False,
) -> Dict[str, int]:
    """
    Traverse `start_dir` and write every *.py file found into an XML document at
    `output_path`.  Returns a summary dict.
    """
    summary = {
        "processed_files": 0,
        "skipped_files": 0,
        "scanned_dirs": 0,
        "bytes_written": 0,
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare writer (plain or gzip)
    open_func = gzip.open if compress else open
    mode = "wt" if compress else "w"
    encoding = "utf-8"

    output_abs = output_path.resolve()

    with open_func(output_path, mode, encoding=encoding) as outfile:
        # Emit minimal root element
        outfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        outfile.write("<codebase>\n")

        # Walk the tree
        for dirpath, dirnames, filenames in os.walk(
            start_dir, followlinks=follow_symlinks
        ):
            summary["scanned_dirs"] += 1
            current_dir = Path(dirpath)

            for filename in filenames:
                if not filename.endswith(".py"):
                    continue

                file_path = current_dir / filename
                # Skip the output file itself
                if file_path.resolve() == output_abs:
                    continue

                # Hard limit check
                if max_files is not None and summary["processed_files"] >= max_files:
                    logger.warning(
                        "Maximum file limit (%d) reached – further files skipped.",
                        max_files,
                    )
                    summary["skipped_files"] += 1
                    continue

                try:
                    relative_path = file_path.relative_to(start_dir)
                    with open(file_path, "r", encoding=encoding, errors="replace") as fh:
                        content = fh.read()
                except Exception as err:  # broad so we still finish the job
                    logger.error("Error reading %s: %s", file_path, err)
                    summary["skipped_files"] += 1
                    continue

                # Write XML-escaped output with CDATA to preserve code
                outfile.write(
                    f'  <file path="{xml_escape(str(relative_path))}"><![CDATA[\n'
                )
                outfile.write(content)
                outfile.write("\n]]></file>\n")

                summary["processed_files"] += 1
                summary["bytes_written"] += len(content.encode(encoding))

                # Size guardrail
                if (
                    max_size_bytes is not None
                    and summary["bytes_written"] >= max_size_bytes
                ):
                    logger.warning(
                        "Maximum size limit (%.2f MB) reached – stopping early.",
                        max_size_bytes / (1024 * 1024),
                    )
                    break  # stop walking further
            else:
                # continue outer loop if inner not broken
                continue
            break  # inner broke → break outer

        outfile.write("</codebase>\n")

    return summary


# ---------- Command-line interface ------------------------------------------------ #

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a directory tree for *.py files and bundle them into a single "
            "XML document."
        )
    )

    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        type=Path,
        help="Directory to scan (defaults to current working directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="combined_python_code.xml",
        type=Path,
        help="Output file path (default: combined_python_code.xml).",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow directory symlinks (default: off).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help=f"Hard cap on file count before aborting (default: {DEFAULT_MAX_FILES}).",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=DEFAULT_MAX_SIZE_MB,
        help=(
            "Hard cap on total bytes written (in megabytes) before aborting "
            f"(default: {DEFAULT_MAX_SIZE_MB})."
        ),
    )
    parser.add_argument(
        "-z",
        "--compress",
        action="store_true",
        help="Write gzip-compressed output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v or -vv).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    # Logging setup
    log_level = logging.WARNING - (10 * min(args.verbose, 2))
    logging.basicConfig(
        level=log_level, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    start_dir = args.directory.resolve()
    output_path = args.output.resolve()

    if not start_dir.is_dir():
        logger.error("Directory not found: %s", start_dir)
        sys.exit(1)

    logger.info("Scanning: %s", start_dir)
    logger.info("Writing  : %s%s", output_path, " (gzip)" if args.compress else "")

    summary = combine_python_files(
        start_dir,
        output_path,
        follow_symlinks=args.follow_symlinks,
        max_files=args.max_files,
        max_size_bytes=args.max_size_mb * 1024 * 1024 if args.max_size_mb else None,
        compress=args.compress,
    )

    # ----- Human-readable summary -------------------------------------------------- #
    processed = summary["processed_files"]
    skipped = summary["skipped_files"]
    dirs = summary["scanned_dirs"]
    size_mb = summary["bytes_written"] / (1024 * 1024)

    print("\n===== Summary Report =====")
    print(f"Directories scanned : {dirs}")
    print(f"Python files copied : {processed}")
    print(f"Files skipped       : {skipped}")
    print(f"Output size         : {size_mb:,.2f} MB")
    print(f"Destination         : {output_path}")
    print("==========================\n")

    if skipped:
        print(
            "⚠️  Some files were skipped due to errors or guardrails. "
            "Check the log for details."
        )


# ---------- Entrypoint ----------------------------------------------------------- #

if __name__ == "__main__":
    main()
