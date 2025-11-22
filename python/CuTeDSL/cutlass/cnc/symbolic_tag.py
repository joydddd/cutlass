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


class SymInt:
    """
    A symbolic integer - either a variable name or a concrete value.
    
    This allows tracking symbolic values through expressions while
    maintaining compatibility with regular Python operations.
    
    Attributes:
        value: Either a string (variable name) or int/None (concrete value)
        is_symbolic: True if this represents a variable name
    
    Examples:
        tile_m = SymInt('tile_m')  # Symbolic
        zero = SymInt(0)           # Concrete
        none = SymInt(None)        # Concrete
        
        # Use in tuples
        tag = (SymInt('tile_m'), SymInt('tile_n'), None)
    """
    
    def __init__(self, value: Union[str, int, None, 'SymInt']):
        """
        Create a symbolic integer.
        
        Args:
            value: Variable name (str), concrete value (int/None), or another SymInt
        """
        if isinstance(value, SymInt):
            self.value = value.value
            self.is_symbolic = value.is_symbolic
        elif isinstance(value, str):
            self.value = value
            self.is_symbolic = True
        else:
            self.value = value
            self.is_symbolic = False
    
    def __repr__(self) -> str:
        if self.is_symbolic:
            return self.value
        else:
            return repr(self.value)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __eq__(self, other) -> bool:
        if isinstance(other, SymInt):
            return self.value == other.value and self.is_symbolic == other.is_symbolic
        return self.value == other
    
    def __hash__(self) -> int:
        return hash((self.value, self.is_symbolic))
    
    def to_ast(self) -> ast.AST:
        """
        Convert to an AST node.
        
        Returns:
            ast.Name for symbolic values, ast.Constant for concrete values
        """
        if self.is_symbolic:
            return ast.Name(id=self.value, ctx=ast.Load())
        elif self.value is None:
            return ast.Constant(value=None)
        else:
            return ast.Constant(value=self.value)
        

# Type alias for symbolic coordinates
# Can be: SymInt, int, None, or Tuple of SymCoords
SymCoord = Union[SymInt, int, None, Tuple['SymCoord', ...]]


def symcoord_to_ast(coord: SymCoord) -> ast.AST:
    """
    Convert any SymCoord-compatible value to an AST node.
    
    Handles: None, int, SymInt, or tuple of SymCoords
    
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