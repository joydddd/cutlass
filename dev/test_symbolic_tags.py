#!/usr/bin/env python3
"""
Test Symbolic Tags

Demonstrates how to use symbolic tags to avoid tag[i] subscripting.
"""

import sys
sys.path.insert(0, '../python/CuTeDSL')

from cutlass.cnc.symbolic_tag import SymbolicTag, make_tag, tag_to_ast
import ast

print("=" * 70)
print("Symbolic Tag System Demo")
print("=" * 70)

# Create symbolic tile indices
tile_m = SymbolicTag('tile_m')
tile_n = SymbolicTag('tile_n')
tile_k = SymbolicTag('tile_k')

print("\n### Scenario: Nested Loop with Evolving Tag ###")
print("\nOuter loop iteration:")
print("  for tile_m, tile_n in cnc.tile(out):")

# Create tag in outer loop
tag = make_tag(tile_m, tile_n, None)
print(f"    tag = {tag}")
print(f"    # Can use: x[{tag[0]}, ?] → x[tile_m, ?]")
print(f"    # Can use: y[?, {tag[1]}] → y[?, tile_n]")

print("\nInner loop iteration:")
print("  for tile_k in cnc.tile(a, mode=[1]):")

# Update tag in inner loop
tag[2] = tile_k
print(f"    tag[2] = tile_k")
print(f"    # Now tag = {tag}")
print(f"    # Can use: x[{tag[0]}, {tag[2]}] → x[tile_m, tile_k]")
print(f"    # Can use: y[{tag[2]}, {tag[1]}] → y[tile_k, tile_n]")

print("\n" + "=" * 70)
print("Code Transformation")
print("=" * 70)

print("\n### Original (with tag subscripting): ###")
original = """
for tile_m, tile_n in cnc.tile(out):
    tag = (tile_m, tile_n, None)
    acc = cnc.zeros_like(out.tile, dtype=torch.float32)
    for tile_k in cnc.tile(a, mode=[1]):
        tag[2] = tile_k
        x_tile = cnc.load(x[tag[0], tag[2]])
        y_tile = cnc.load(y[tag[2], tag[1]])
        acc = cnc.addmm(acc, x_tile, y_tile)
    cnc.store(acc, out[tag[0], tag[1]])
"""
print(original)

print("\n### With Symbolic Tags (cleaner): ###")
symbolic = """
# Create symbolic tags
tile_m = SymbolicTag('tile_m')
tile_n = SymbolicTag('tile_n')
tile_k = SymbolicTag('tile_k')

for tile_m_val, tile_n_val in cnc.tile(out):
    # Bind symbolic tags to actual values
    tile_m.bind(tile_m_val)
    tile_n.bind(tile_n_val)
    
    acc = cnc.zeros_like(out.tile, dtype=torch.float32)
    for tile_k_val in cnc.tile(a, mode=[1]):
        tile_k.bind(tile_k_val)
        
        # Direct use - no subscripting needed!
        x_tile = cnc.load(x[tile_m, tile_k])
        y_tile = cnc.load(y[tile_k, tile_n])
        acc = cnc.addmm(acc, x_tile, y_tile)
    
    cnc.store(acc, out[tile_m, tile_n])
"""
print(symbolic)

print("\n" + "=" * 70)
print("AST Generation")
print("=" * 70)

# Show how to generate AST from symbolic tags
print("\n### Generate AST nodes: ###")

# Outer tag
outer_tag = make_tag('tile_m', 'tile_n', None)
print(f"\nOuter tag: {outer_tag}")
print(f"As code: {outer_tag.to_code()}")
print(f"As AST: {ast.dump(outer_tag.to_ast())}")

# Inner tag (after update)
inner_tag = make_tag('tile_m', 'tile_n', 'tile_k')
print(f"\nInner tag: {inner_tag}")
print(f"As code: {inner_tag.to_code()}")

# Create assignment statements
outer_assign = ast.Assign(
    targets=[ast.Name(id='tag', ctx=ast.Store())],
    value=outer_tag.to_ast()
)
ast.fix_missing_locations(outer_assign)
print(f"\nOuter assignment: {ast.unparse(outer_assign)}")

inner_assign = ast.Assign(
    targets=[ast.Name(id='tag', ctx=ast.Store())],
    value=inner_tag.to_ast()
)
ast.fix_missing_locations(inner_assign)
print(f"Inner assignment: {ast.unparse(inner_assign)}")

print("\n" + "=" * 70)
print("Runtime Tag Evolution")
print("=" * 70)

# Simulate runtime behavior
print("\n### Simulating nested loops: ###")

# Outer loop
outer_tag = make_tag('tile_m', 'tile_n', None)
print(f"\n1. Outer loop creates: tag = {outer_tag}")

# Inner loop updates
inner_tag = outer_tag.copy()  # Copy to simulate scope
inner_tag[2] = 'tile_k'
print(f"2. Inner loop updates: tag[2] = tile_k")
print(f"3. Inner loop sees: tag = {inner_tag}")

# Show what each sees
print(f"\n4. Usage in inner loop:")
print(f"   x[tag[0], tag[2]] → x[{inner_tag[0]}, {inner_tag[2]}]")
print(f"   y[tag[2], tag[1]] → y[{inner_tag[2]}, {inner_tag[1]}]")

# After inner loop
print(f"\n5. After inner loop (in outer scope):")
print(f"   out[tag[0], tag[1]] → out[{outer_tag[0]}, {outer_tag[1]}]")

print("\n" + "=" * 70)
print("Integration with AST Graph")
print("=" * 70)

print("\n### How to use in your AST nodes: ###")

code_example = '''
import cutlass.cnc.ast as cnc_ast
from cutlass.cnc.symbolic_tag import make_tag

# Create symbolic tags
outer_tag = make_tag('tile_m', 'tile_n', None)
inner_tag = make_tag('tile_m', 'tile_n', 'tile_k')

# Use in _new_tag
outer_loop = cnc_ast.create(
    ast.For,
    _visualize=True,
    _new_tag=ast.Assign(
        targets=[ast.Name(id='tag', ctx=ast.Store())],
        value=outer_tag.to_ast()  # Convert to AST!
    ),
    ...
)

inner_loop = cnc_ast.create(
    ast.For,
    _visualize=True,
    _new_tag=ast.Assign(
        targets=[ast.Name(id='tag', ctx=ast.Store())],
        value=inner_tag.to_ast()  # Convert to AST!
    ),
    ...
)
'''
print(code_example)

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Symbolic Tags provide:
  ✓ Runtime tracking of symbolic indices
  ✓ Easy conversion to AST (tag.to_ast())
  ✓ Readable code (no tag[i] subscripting)
  ✓ Proper scope handling (copy for nested scopes)
  ✓ Compatible with cute::Coord design philosophy
  
Use Cases:
  • Creating AST nodes with inferred tags
  • Tracking tag evolution through nested loops
  • Generating visualization-ready code
  • Avoiding manual AST transformation
""")

