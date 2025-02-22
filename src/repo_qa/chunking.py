import os
import ast
from pathlib import Path

from .config import SystemConfig

def extract_code_blocks(repo_path: str, chunk_size: int = 100):
    """
    Walks through all files in repo_path and yields (chunk_text, metadata).
    - For Python files: splits by function/class definitions
    - For non-Python files: splits into fixed-size chunks
    """
    repo_path = Path(repo_path)
    for file in repo_path.rglob("*"):
        if not file.is_file() or os.stat(file).st_size == 0:
            continue
        elif file.suffix in SystemConfig.file_suffixes:
            try:
                with open(file, "r") as f:
                    source = f.read()
                if file.suffix == ".py":
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            start_line = node.lineno
                            end_line = getattr(node, 'end_lineno', None)
                            if end_line is None:
                                end_line = find_end_line(node)

                            code_lines = source.split("\n")[start_line-1:end_line]
                            code_block = "\n".join(code_lines)

                            metadata = {
                                "file_path": str(file),
                                "name": node.name,
                                "block_type": (
                                    "class" if isinstance(node, ast.ClassDef) else "function"
                                ),
                                "start_line": start_line,
                                "end_line": end_line
                            }
                            yield code_block, metadata
                else:
                    # Handle non-Python files by splitting into fixed-size chunks
                    lines = source.split("\n")
                    for i in range(0, len(lines), chunk_size):
                        chunk_lines = lines[i:i + chunk_size]
                        chunk = "\n".join(chunk_lines)
                        metadata = {
                            "file_path": str(file),
                            "name": f"chunk_{i//chunk_size + 1}",
                            "block_type": "text",
                            "start_line": i + 1,
                            "end_line": min(i + chunk_size, len(lines))
                        }
                        yield chunk, metadata
            except Exception as e:
                print(f"Skipping file {file}")

def find_end_line(node):
    """
    Fallback function to estimate the end line if 'end_lineno' is unavailable.
    We can do a DFS in node.body or otherwise approximate.
    """
    max_line = node.lineno
    for child in ast.walk(node):
        if hasattr(child, 'lineno'):
            max_line = max(max_line, child.lineno)
    return max_line