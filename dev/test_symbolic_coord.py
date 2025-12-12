#!/usr/bin/env python3
"""
Test Symbolic Coordinates (cute::Coord-like)

Demonstrates the symbolic coordinate system that mimics cute::Coord behavior.
"""

import sys
sys.path.insert(0, '../python/CuTeDSL')

from cutlass.cnc.symbolic_tag import SymbolicCoord, make_coord, coord_to_ast
import ast

print("=" * 70)
print("Symbolic Coordinate System (cute::Coord-like)")
print("=" * 70)

print("\n### Mimics cute::Coord Interface ###")
print("\nIn C++:")
print("  auto coord = cute::make_coord(tile_m, tile_n, tile_k);")
print("  coord[0]  // Access first element")
print("  coord.at(1)  // Access second element")
print("\nIn Python (Symbolic):")
print("  coord = make_coord('tile_m', 'tile_n', 'tile_k')")
print("  coord[0]  # Access first element")
print("  coord.at(1)  # Access second element")

# Create coordinate
coord = make_coord('tile_m', 'tile_n', 'tile_k')
print(f"\ncoord = {coord}")
print(f"rank = {coord.rank}")
print(f"coord[0] = {coord[0]}")
print(f"coord.at(1) = {coord.at(1)}")
print(f"coord[2] = {coord[2]}")

print("\n" + "=" * 70)
print("Nested Loop Pattern (your use case)")
print("=" * 70)

print("\n### Outer Loop ###")
print("for tile_m, tile_n in cnc.tile(out):")

# Create outer coordinate
outer_tag = make_coord('tile_m', 'tile_n', None)
print(f"  tag = {outer_tag}")
print(f"  # rank: {outer_tag.rank}")

print("\n### Inner Loop ###")
print("  for tile_k in cnc.tile(a, mode=[1]):")

# Copy and update for inner scope
inner_tag = outer_tag.copy()
inner_tag[2] = 'tile_k'
print(f"    tag[2] = 'tile_k'")
print(f"    # Now tag = {inner_tag}")

print("\n### Generated Code ###")
print(f"Outer scope: tag = {outer_tag.to_code()}")
print(f"Inner scope: tag = {inner_tag.to_code()}")

print("\n" + "=" * 70)
print("AST Generation")
print("=" * 70)

# Create AST nodes for visualization
print("\n### For _new_tag in AST nodes ###")

outer_assign = ast.Assign(
    targets=[ast.Name(id='tag', ctx=ast.Store())],
    value=outer_tag.to_ast()
)
ast.fix_missing_locations(outer_assign)

inner_assign = ast.Assign(
    targets=[ast.Name(id='tag', ctx=ast.Store())],
    value=inner_tag.to_ast()
)
ast.fix_missing_locations(inner_assign)

print(f"Outer: {ast.unparse(outer_assign)}")
print(f"Inner: {ast.unparse(inner_assign)}")

print("\n### Integration with cnc_ast ###")
code = '''
import cutlass.cnc.ast as cnc_ast
from cutlass.cnc.symbolic_tag import make_coord

# Create coordinates
outer_tag = make_coord('tile_m', 'tile_n', None)
inner_tag = make_coord('tile_m', 'tile_n', 'tile_k')

# Use in AST nodes
outer_loop = cnc_ast.create(
    ast.For,
    _new_tag=ast.Assign(
        targets=[ast.Name(id='tag', ctx=ast.Store())],
        value=outer_tag.to_ast()
    ),
    ...
)

inner_loop = cnc_ast.create(
    ast.For,
    _new_tag=ast.Assign(
        targets=[ast.Name(id='tag', ctx=ast.Store())],
        value=inner_tag.to_ast()
    ),
    ...
)
'''
print(code)

print("\n" + "=" * 70)
print("Advanced Features")
print("=" * 70)

print("\n### 1. Nested Coordinates ###")
inner = make_coord('i', 'j')
outer = make_coord('m', 'n', inner)
print(f"inner = {inner}")
print(f"outer = {outer}")
print(f"outer[2] = {outer[2]}")
print(f"As code: {outer.to_code()}")

print("\n### 2. Mixed Symbolic/Concrete ###")
mixed = make_coord('tile_m', 0, None, 'tile_k')
print(f"mixed = {mixed}")
print(f"As code: {mixed.to_code()}")

print("\n### 3. Slicing ###")
coord = make_coord('a', 'b', 'c', 'd')
print(f"coord = {coord}")
print(f"coord[1:3] = {coord[1:3]}")

print("\n### 4. Iteration ###")
for i, elem in enumerate(coord):
    print(f"  coord[{i}] = {elem}")

print("\n### 5. Conversion to Python tuple ###")
coord = make_coord('x', 'y', 'z')
print(f"coord = {coord}")
print(f"as_tuple() = {coord.as_tuple()}")

print("\n" + "=" * 70)
print("Comparison with Original Code")
print("=" * 70)

print("\n### Original (with subscript indexing): ###")
print("""
for tile_m, tile_n in cnc.tile(out):
    tag = (tile_m, tile_n, None)
    for tile_k in cnc.tile(a, mode=[1]):
        tag[2] = tile_k
        x_tile = cnc.load(x[tag[0], tag[2]])  # tag[0], tag[2]
        y_tile = cnc.load(y[tag[2], tag[1]])  # tag[2], tag[1]
""")

print("\n### With SymbolicCoord (for AST generation): ###")
print("""
from cutlass.cnc.symbolic_tag import make_coord

# Define tag structure symbolically
outer_tag = make_coord('tile_m', 'tile_n', None)
inner_tag = make_coord('tile_m', 'tile_n', 'tile_k')

# Use these to create AST nodes with proper _new_tag
outer_loop = cnc_ast.create(
    ast.For,
    _new_tag=ast.Assign(..., value=outer_tag.to_ast()),
    ...
)

# The visualization will show: tag = (tile_m, tile_n, tile_k)
""")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
SymbolicCoord features:
  ✓ Mimics cute::Coord interface (rank, [], at())
  ✓ Supports symbolic values (variable names as strings)
  ✓ Supports nested coordinates
  ✓ Supports mixed symbolic/concrete values
  ✓ Easy conversion to AST (to_ast())
  ✓ Clean code generation (to_code())
  
Key methods:
  • make_coord(...) - Create coordinate
  • coord[i] - Access element
  • coord[i] = val - Set element
  • coord.rank - Get dimensionality
  • coord.at(i) - Access element (explicit)
  • coord.copy() - Deep copy
  • coord.to_ast() - Convert to AST
  • coord.to_code() - Convert to code string
  
Use cases:
  • Track tag evolution through nested loops
  • Generate AST for visualization
  • Avoid manual AST construction
""")

