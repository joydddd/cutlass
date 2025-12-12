#!/usr/bin/env python3
"""
Test Symbolic Integer

Simple symbolic integers that work with regular Python tuples.
"""

import sys
sys.path.insert(0, '../python/CuTeDSL')

from cutlass.cnc.symbolic_tag import SymbolicInt, tuple_to_ast, tuple_to_code
import ast

print("=" * 70)
print("Symbolic Integer - Simple Approach")
print("=" * 70)

print("\n### Core Idea: Just use SymbolicInt with Python tuples ###")

# Create symbolic integers
tile_m = SymbolicInt('tile_m')
tile_n = SymbolicInt('tile_n')
tile_k = SymbolicInt('tile_k')

print(f"\ntile_m = {tile_m}")
print(f"tile_n = {tile_n}")
print(f"tile_k = {tile_k}")

print("\n" + "=" * 70)
print("Usage with Regular Python Tuples")
print("=" * 70)

print("\n### Create tag as regular tuple ###")
tag = (tile_m, tile_n, None)
print(f"tag = {tag}")
print(f"tag[0] = {tag[0]}")
print(f"tag[1] = {tag[1]}")
print(f"tag[2] = {tag[2]}")

print("\n### Update element (convert to list, modify, back to tuple) ###")
tag_list = list(tag)
tag_list[2] = tile_k
tag = tuple(tag_list)
print(f"tag = {tag}")

print("\n### Convert to AST ###")
print(f"As code: {tuple_to_code(tag)}")
print(f"As AST: {ast.dump(tuple_to_ast(tag))}")

print("\n" + "=" * 70)
print("Nested Loop Scenario")
print("=" * 70)

print("\n### Outer loop ###")
outer_tag = (SymbolicInt('tile_m'), SymbolicInt('tile_n'), None)
print(f"tag = {tuple_to_code(outer_tag)}")

print("\n### Inner loop (copy and update) ###")
inner_tag = list(outer_tag)
inner_tag[2] = SymbolicInt('tile_k')
inner_tag = tuple(inner_tag)
print(f"tag = {tuple_to_code(inner_tag)}")

print("\n### Create AST assignments ###")
outer_assign = ast.Assign(
    targets=[ast.Name(id='tag', ctx=ast.Store())],
    value=tuple_to_ast(outer_tag)
)
ast.fix_missing_locations(outer_assign)

inner_assign = ast.Assign(
    targets=[ast.Name(id='tag', ctx=ast.Store())],
    value=tuple_to_ast(inner_tag)
)
ast.fix_missing_locations(inner_assign)

print(f"Outer: {ast.unparse(outer_assign)}")
print(f"Inner: {ast.unparse(inner_assign)}")

print("\n" + "=" * 70)
print("Shortcut: Use Bare Strings")
print("=" * 70)

print("\n### Even simpler - bare strings work too! ###")
tag = ('tile_m', 'tile_n', None)
print(f"tag = {tag}")
print(f"As code: {tuple_to_code(tag)}")

# Update
tag_list = list(tag)
tag_list[2] = 'tile_k'
tag = tuple(tag_list)
print(f"Updated: {tuple_to_code(tag)}")

print("\n" + "=" * 70)
print("Integration Example")
print("=" * 70)

code = '''
import cutlass.cnc.ast as cnc_ast
from cutlass.cnc.symbolic_tag import tuple_to_ast

# Define tags as simple tuples
outer_tag = ('tile_m', 'tile_n', None)
inner_tag = ('tile_m', 'tile_n', 'tile_k')

# Use in AST nodes
outer_loop = cnc_ast.create(
    ast.For,
    _new_tag=ast.Assign(
        targets=[ast.Name(id='tag', ctx=ast.Store())],
        value=tuple_to_ast(outer_tag)  # Just convert tuple to AST!
    ),
    ...
)

inner_loop = cnc_ast.create(
    ast.For,
    _new_tag=ast.Assign(
        targets=[ast.Name(id='tag', ctx=ast.Store())],
        value=tuple_to_ast(inner_tag)
    ),
    ...
)
'''
print(code)

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Advantages of SymbolicInt-only approach:
  ✓ Use regular Python tuples
  ✓ Simple and intuitive
  ✓ No custom container class needed
  ✓ Standard tuple operations work
  ✓ Easy conversion to AST with tuple_to_ast()
  
Usage:
  1. Create symbolic ints: tile_m = SymbolicInt('tile_m')
  2. Use in tuples: tag = (tile_m, tile_n, None)
  3. Update: tag = list(tag); tag[2] = tile_k; tag = tuple(tag)
  4. Convert: tuple_to_ast(tag)
  
Or even simpler with strings:
  1. tag = ('tile_m', 'tile_n', None)
  2. Convert: tuple_to_ast(tag)  # Strings treated as symbolic!
""")

