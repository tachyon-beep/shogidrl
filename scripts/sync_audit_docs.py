import os
import ast
import datetime
import re
from pathlib import Path

TEMPLATE_PATH = Path('docs/templates/SUBSYSTEM_TEMPLATE.md')
OUTPUT_DIR = Path('docs/component_audit')

with TEMPLATE_PATH.open('r', encoding='utf-8') as f:
    TEMPLATE_TEXT = f.read()

HEADER_PROJECT = 'Keisei - Deep Reinforcement Learning Shogi Client'
HEADER_VERSION = '1.0'
HEADER_AUTHOR = 'GitHub Copilot'

def extract_info(py_path: Path):
    source = py_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree)
    missing_docstring = docstring is None
    if docstring:
        docstring = docstring.strip().splitlines()[0]
    else:
        docstring = 'No docstring found—please add a summary'

    classes = []
    functions = []
    imports = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            doc = doc.strip().splitlines()[0] if doc else 'No docstring'
            classes.append((node.name, doc))
        elif isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            doc = doc.strip().splitlines()[0] if doc else 'No docstring'
            functions.append((node.name, doc))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            line = source.splitlines()[node.lineno-1].strip()
            imports.append(line)

    todos = []
    for i, line in enumerate(source.splitlines(), start=1):
        if 'TODO' in line or 'FIXME' in line:
            todos.append(f"Line {i}: {line.strip()}")

    return {
        'docstring': docstring,
        'classes': classes,
        'functions': functions,
        'imports': imports,
        'todos': todos,
        'missing_docstring': missing_docstring,
    }

def replace_section(text: str, heading_pattern: str, new_content: str) -> str:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if re.match(heading_pattern, line):
            start = idx
            break
    if start is None:
        return text
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if re.match(r'^### ', lines[idx]):
            end = idx
            break
    new_lines = [lines[start], ''] + new_content.splitlines()
    lines[start:end] = new_lines
    return '\n'.join(lines)

def build_doc(info, module_path: Path) -> str:
    today = datetime.date.today().isoformat()
    module_name = module_path.stem
    folder_path = '/' + module_path.parent.as_posix() + '/'

    text = TEMPLATE_TEXT
    text = text.replace('[MODULE NAME]', module_name)
    text = text.replace('[FILENAME]', module_path.name)
    text = text.replace('[DATE]', today)
    text = text.replace('[Insert Project Name]', HEADER_PROJECT)
    text = text.replace('[Insert Full Folder Path]', folder_path)
    text = text.replace('[Insert Version]', HEADER_VERSION)
    text = text.replace('[Insert Date]', today)
    text = text.replace('[Insert if applicable]', HEADER_AUTHOR)

    overview = info['docstring']
    classes = '\n'.join(f"- `{n}`: {d}" for n, d in info['classes']) or '*None*'
    functions = '\n'.join(f"- `{n}`: {d}" for n, d in info['functions']) or '*None*'
    imports = '\n'.join(f"- {l}" for l in info['imports']) or '*None*'
    todos = '\n'.join(f"- {t}" for t in info['todos']) or '*No TODO/FIXME comments found*'

    text = replace_section(text, r'^### 1\. Overview', overview)
    text = replace_section(text, r'^### 3\. Classes', classes)
    text = replace_section(text, r'^### 4\. Functions', functions)
    text = replace_section(text, r'^### 2\. Modules', f"* **Dependencies:**\n{imports}")
    text = replace_section(text, r'^### 10\. Known Issues', todos)

    return text

def main():
    modules = sorted(Path('keisei').rglob('*.py'))
    created = 0
    overwritten = 0
    warnings = []
    for module in modules:
        info = extract_info(module)
        doc_path = OUTPUT_DIR / module.as_posix().replace('/', '_').replace('.py', '.md')
        doc_text = build_doc(info, module)
        if not doc_path.exists():
            created += 1
            action = 'Created'
        else:
            overwritten += 1
            action = 'Overwrote'
        doc_path.write_text(doc_text, encoding='utf-8')
        print(f"[✓] {action} {doc_path}")
        if info['missing_docstring']:
            warnings.append(f"[!] Warning: {module} has no module docstring—insert 'No docstring found' placeholder.")
    print(f"Processed {len(modules)} modules. Docs created: {created}, overwritten: {overwritten}.")
    if warnings:
        print('\n'.join(warnings))

if __name__ == '__main__':
    main()
