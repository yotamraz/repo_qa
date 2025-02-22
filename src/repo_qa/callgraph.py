
import ast
from collections import defaultdict
from pathlib import Path

class CallGraphBuilder(ast.NodeVisitor):
    """

    """
    def __init__(self):
        self.current_func = None # keep track of last visited func
        self.graph = defaultdict(set)  # e.g. {"functionA": {"functionB", "functionC"}}
        self.defined_funcs = set()  # hold all function names we identify

    def visit_FunctionDef(self, node):
        # Record that we have a function named node.name
        old_func = self.current_func
        self.current_func = node.name
        self.defined_funcs.add(node.name)

        # Continue walking the function body
        self.generic_visit(node)
        # Restore
        self.current_func = old_func

    def visit_AsyncFunctionDef(self, node):
        # treat same as function for simplicity
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        # We can treat classes as well, if we want class-level calls or methods
        old_func = self.current_func
        self.current_func = node.name
        self.defined_funcs.add(node.name)

        self.generic_visit(node)
        self.current_func = old_func

    def visit_Call(self, node):
        # If the function name is directly a Name node
        if isinstance(node.func, ast.Name):
            called_name = node.func.id
            if self.current_func and called_name:
                self.graph[self.current_func].add(called_name)
        # If it's an Attribute, e.g., self.some_method or module.function
        elif isinstance(node.func, ast.Attribute):
            # For simplicity, just store the final attribute name
            called_name = node.func.attr
            if self.current_func and called_name:
                self.graph[self.current_func].add(called_name)
        # Continue traversing
        self.generic_visit(node)


def build_call_graph(repo_path: str):
    """
    Parse all .py files in repo_path, build a global call graph
    { caller_name -> set of callee_names }
    """
    builder = CallGraphBuilder()
    repo_path = Path(repo_path)
    for py_file in repo_path.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            builder.visit(tree)
        except Exception as e:
            print(f"Failed on {py_file}: {e}")

    return dict(builder.graph), builder.defined_funcs
