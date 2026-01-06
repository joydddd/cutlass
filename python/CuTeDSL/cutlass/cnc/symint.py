"""
Symbolic Integer

A simple symbolic integer that can represent either:
- A variable name (symbolic): 'tile_m', 'tile_n', 'tile_k'
- A concrete value: 0, 1, 2, None

Use regular Python tuples for coordinates:
    tag = (SymInt('tile_m'), SymInt('tile_n'), None)
"""
from __future__ import annotations
import ast
from typing import Union, Optional, Tuple

from dataclasses import dataclass
import sympy


# Type alias for symbolic coordinates
# Can be: SymInt, int, None, or Tuple of SymCoords
SymCoord = Union["SymInt", int, None, Tuple['SymCoord', ...]]
SymTile = Union[sympy.Symbol, Tuple[sympy.Symbol, ...]]


@dataclass(frozen=True)
class SymInt:
    """
    Symbolic integer backed by a SymPy expression.

    Can be:
      - a variable name: SymInt("tile_m")  -> Symbol("tile_m")
      - a concrete int:  SymInt(4)        -> Integer(4)
      - any expression built from other SymInt/ints via +, -, *, //, %, **, unary -
    """
    expr: sympy.Expr

    def __init__(self, value: Union[int, str, sympy.Expr, "SymInt"]) -> None:
        if isinstance(value, SymInt):
            expr = value.expr
        elif isinstance(value, sympy.Expr):
            expr = value
        elif isinstance(value, int):
            expr = sympy.Integer(value)
        elif isinstance(value, str):
            expr = sympy.Symbol(value)
        else:
            raise TypeError(f"Unsupported value for SymInt: {type(value)}")

        object.__setattr__(self, "expr", expr)

    # ---- helpers ----

    @staticmethod
    def _coerce(other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        if isinstance(other, SymInt):
            return other
        if isinstance(other, sympy.Expr):
            return SymInt(other)
        if isinstance(other, int):
            return SymInt(other)
        raise TypeError(f"Cannot coerce {type(other)} to SymInt")

    # ---- arithmetic (delegates to SymPy) ----

    def __add__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self.expr + self._coerce(other).expr)

    def __radd__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self._coerce(other).expr + self.expr)

    def __sub__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self.expr - self._coerce(other).expr)

    def __rsub__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self._coerce(other).expr - self.expr)

    def __mul__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self.expr * self._coerce(other).expr)

    def __rmul__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self._coerce(other).expr * self.expr)

    def __floordiv__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        # SymPy uses /; you can wrap in floor() if you want true floor semantics
        return SymInt(sp.floor(self.expr / self._coerce(other).expr))

    def __rfloordiv__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(sp.floor(self._coerce(other).expr / self.expr))

    def __mod__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self.expr % self._coerce(other).expr)

    def __rmod__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self._coerce(other).expr % self.expr)

    def __pow__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self.expr ** self._coerce(other).expr)

    def __rpow__(self, other: Union[int, "SymInt", sympy.Expr]) -> "SymInt":
        return SymInt(self._coerce(other).expr ** self.expr)

    def __neg__(self) -> "SymInt":
        return SymInt(-self.expr)

    # ---- conversions / debugging ----

    def to_sympy(self) -> sympy.Expr:
        """Return the underlying SymPy expression."""
        return self.expr

    def to_ast(self) -> ast.AST:
        """
        Convert to a Python AST expression node by unparsing via SymPy's string form.
        Useful if the rest of your pipeline still wants an ast.AST.
        """
        src = str(self.expr)
        return ast.parse(src, mode="eval").body

    def __repr__(self) -> str:
        return f"SymInt({self.expr})"


def symcoord_to_ast(coord: SymCoord) -> ast.AST:
    """
    Convert any SymCoord-compatible value to an AST node.
    
    Handles: None, int, SymInt, or tuple of SymCoord
    
    Args:
        coord: A SymCoord-compatible value
        
    Returns:
        ast.AST node representing the coordinate
        
    Examples:
        symcoord_to_ast(None)                    # -> ast.Constant(None)
        symcoord_to_ast(5)                       # -> ast.Constant(5)
        symcoord_to_ast(SymInt('x'))             # -> ast.Name('x')
        symcoord_to_ast((SymInt('i'), None))     # -> ast.Tuple([...])
    """
    if coord is None:
        return ast.Constant(value=None)
    
    if isinstance(coord, int):
        return ast.Constant(value=coord)
    
    if isinstance(coord, SymInt):
        return coord.to_ast()
    
    if isinstance(coord, tuple):
        # Tuple - convert each element recursively
        elements = [symcoord_to_ast(item) for item in coord]
        return ast.Tuple(elts=elements, ctx=ast.Load())
    
    # Fallback
    return ast.Constant(value=coord)


class SymIntSource:
    _prefix: str
    _count: int = 0
    def __init__(self, prefix: str = "coord"):
        self._prefix = prefix
        self._count = 0

    def create_new(self) -> SymInt:
        symint = SymInt(f"{self._prefix}{self._count}")
        self._count += 1
        return symint
        