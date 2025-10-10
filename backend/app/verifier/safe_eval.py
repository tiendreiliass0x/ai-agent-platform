from __future__ import annotations

import ast
from typing import Any, Dict

from .exceptions import ConditionEvaluationError

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,  # limited to allowed functions
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
)

_ALLOWED_FUNCS = {
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
}


class SafeEvaluator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.allowed_funcs = _ALLOWED_FUNCS

    def visit(self, node):  # type: ignore[override]
        if not isinstance(node, _ALLOWED_NODES):
            raise ConditionEvaluationError(f"Disallowed expression: {ast.dump(node)}")
        return super().visit(node)

    def visit_Call(self, node: ast.Call):  # noqa: D401
        if not isinstance(node.func, ast.Name):
            raise ConditionEvaluationError("Only simple function calls allowed")
        if node.func.id not in self.allowed_funcs:
            raise ConditionEvaluationError(f"Function '{node.func.id}' not permitted")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node: ast.Attribute):  # noqa: D401
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript):  # noqa: D401
        self.visit(node.value)
        self.visit(node.slice)

    def visit_List(self, node: ast.List):  # noqa: D401
        for elt in node.elts:
            self.visit(elt)

    def visit_Tuple(self, node: ast.Tuple):  # noqa: D401
        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node: ast.Dict):  # noqa: D401
        for key in node.keys:
            if key is not None:
                self.visit(key)
        for value in node.values:
            self.visit(value)


def safe_eval(expression: str, context: Dict[str, Any]) -> Any:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ConditionEvaluationError(f"Invalid expression: {expression}") from exc

    SafeEvaluator().visit(parsed)
    compiled = compile(parsed, filename="<condition>", mode="eval")

    safe_globals = {"__builtins__": {}, **_ALLOWED_FUNCS}
    try:
        return eval(compiled, safe_globals, context)
    except Exception as exc:
        raise ConditionEvaluationError(f"Error evaluating expression '{expression}': {exc}") from exc
