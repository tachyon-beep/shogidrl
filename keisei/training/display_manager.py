"""
training/display_manager.py: Manages Rich UI display components and logging.
"""

import sys
from typing import List, Optional, Any

from rich.console import Console, Text
from rich.live import Live

from . import display


class DisplayManager:
    """Manages Rich console display, logging, and UI components."""

    def __init__(self, config: Any, log_file_path: str):
        """
        Initialize the DisplayManager.

        Args:
            config: Training configuration
            log_file_path: Path to the log file
        """
        self.config = config
        self.log_file_path = log_file_path
        
        # Initialize Rich console components
        self.rich_console = Console(file=sys.stderr, record=True)
        self.rich_log_messages: List[Text] = []
        
        # Display will be initialized when needed
        self.display: Optional[display.TrainingDisplay] = None

    def setup_display(self, trainer) -> display.TrainingDisplay:
        """
        Setup the training display.

        Args:
            trainer: The trainer instance

        Returns:
            The configured TrainingDisplay
        """
        self.display = display.TrainingDisplay(
            self.config, trainer, self.rich_console
        )
        return self.display

    def get_console(self) -> Console:
        """
        Get the Rich console instance.

        Returns:
            The Rich console
        """
        return self.rich_console

    def get_log_messages(self) -> List[Text]:
        """
        Get the log messages list.

        Returns:
            List of Rich Text log messages
        """
        return self.rich_log_messages

    def add_log_message(self, message: str) -> None:
        """
        Add a log message to the Rich log panel.

        Args:
            message: The message to add
        """
        rich_message = Text(message)
        self.rich_log_messages.append(rich_message)

    def update_progress(self, trainer, speed: float, pending_updates: dict) -> None:
        """
        Update the progress display.

        Args:
            trainer: The trainer instance
            speed: Training speed in steps/second
            pending_updates: Dictionary of pending progress updates
        """
        if self.display:
            self.display.update_progress(trainer, speed, pending_updates)

    def update_log_panel(self, trainer) -> None:
        """
        Update the log panel display.

        Args:
            trainer: The trainer instance
        """
        if self.display:
            self.display.update_log_panel(trainer)

    def start_live_display(self) -> Optional[Live]:
        """
        Start the Rich Live display context manager.

        Returns:
            Live context manager if display is available, None otherwise
        """
        if self.display:
            return self.display.start()
        return None

    def save_console_output(self, output_dir: str) -> bool:
        """
        Save the Rich console output to HTML.

        Args:
            output_dir: Directory to save the output

        Returns:
            True if successful, False otherwise
        """
        try:
            import os
            console_log_path = os.path.join(output_dir, "full_console_output_rich.html")
            self.rich_console.save_html(console_log_path)
            print(
                f"Full Rich console output saved to {console_log_path}", 
                file=sys.stderr
            )
            return True
        except OSError as e:
            print(f"Error saving Rich console log: {e}", file=sys.stderr)
            return False

    def print_rule(self, title: str, style: str = "bold green") -> None:
        """
        Print a rule with title to the console.

        Args:
            title: The title for the rule
            style: Style for the rule
        """
        self.rich_console.rule(f"[{style}]{title}[/{style}]")

    def print_message(self, message: str, style: str = "bold green") -> None:
        """
        Print a styled message to the console.

        Args:
            message: The message to print
            style: Style for the message
        """
        self.rich_console.print(f"[{style}]{message}[/{style}]")

    def finalize_display(self, run_name: str, run_artifact_dir: str) -> None:
        """
        Finalize the display and save console output.

        Args:
            run_name: Name of the training run
            run_artifact_dir: Directory for run artifacts
        """
        # Save console output
        self.save_console_output(run_artifact_dir)
        
        # Final messages
        self.print_rule("Run Finished")
        self.print_message(f"Run '{run_name}' processing finished.")
        self.print_message(f"Output and logs are in: {run_artifact_dir}")
