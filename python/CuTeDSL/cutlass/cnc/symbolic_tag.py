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
from torch import TorchSymInt


# Type alias for symbolic coordinates
# Can be: SymInt, int, None, or Tuple of SymCoords
SymCoord = Union[TorchSymInt, int, None, Tuple['SymCoord', ...]]


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
    
    if isinstance(coord, TorchSymInt):
        return coord.to_ast()
    
    if isinstance(coord, tuple):
        # Tuple - convert each element recursively
        elements = [symcoord_to_ast(item) for item in coord]
        return ast.Tuple(elts=elements, ctx=ast.Load())
    
    # Fallback
    return ast.Constant(value=coord)