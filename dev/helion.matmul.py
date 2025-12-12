import ast


import cutlass.cnc.ast as cnc_ast
from cutlass.cnc.scheduler import tag_push

from cutlass.cnc.symbolic_tag import SymCoord, symcoord_to_ast


from torch import SymInt


from helion._compiler.ast_extension import statement_from_string


### Node 0: The grid loop. 
def create_grid_loop_node(tag: SymCoord) -> cnc_ast.AstExtension:
    iter_node = ast.parse("cnc.tile(out_paritioner, mode=[0, 1])", mode="eval").body


    target = (SymInt("coord_m"), SymInt("coord_n"))
    new_tag = tag_push(tag, target)

    body_nodes = [
        create_init_node(new_tag),
        create_inner_loop_node(new_tag),
        create_store_node(new_tag)
    ]

    return cnc_ast.create(
        ast.For,  
        _visualize=True,
        _tag = tag, 
        _new_tag = new_tag,
        target=symcoord_to_ast(target),
        iter=iter_node,
        body=body_nodes, 
        orelse=[],
        type_comment=None
    )



### Node 1: The init node. 
def create_init_node(tag: SymCoord) -> cnc_ast.AstExtension:

    output = ast.Name(id="acc", ctx=ast.Store())
    init_node = statement_from_string("{output} = cnc.zeros_like(out_paritioner.tile, dtype=torch.float32)", output=output)

    return cnc_ast.create(
        ast.Module, 
        _tag = tag,
        _visualize=True,
        _inputs=[], 
        _outputs=[output],
        body=[init_node]
    )

### Node 2: The inner loop node, 
def create_inner_loop_node(tag: SymCoord) -> cnc_ast.AstExtension:
    iter_node = ast.parse(f"cnc.tile(a_paritioner, mode=[1])", mode="eval").body
    target = SymInt("coord_k")
    new_tag = tag_push(tag, target)

    return cnc_ast.create(
        ast.For,  
        _visualize=True,
        _tag = tag,
        _new_tag = new_tag,
        target=symcoord_to_ast(target),
        iter=iter_node,
        body=[
            create_load_xy(new_tag),
            create_mma(new_tag), 
        ], 
        orelse=[],
        type_comment=None
    )


### Node 3: create load node
def create_load_xy(tag: SymCoord) -> cnc_ast.AstExtension:
    x_tile_node = ast.Name(id="x_tile", ctx=ast.Store())
    y_tile_node = ast.Name(id="y_tile", ctx=ast.Store())
    x_sub = tag[0][0], tag[1]
    y_sub = tag[1], tag[0][1]
    x_load = statement_from_string("{x_tile} = cnc.load(x[{x_sub}])", x_sub=symcoord_to_ast(x_sub), x_tile= x_tile_node)
    y_load = statement_from_string("{y_tile} = cnc.load(y[{y_sub}])", y_sub=symcoord_to_ast(y_sub), y_tile= y_tile_node)

    return cnc_ast.create(
        ast.Module, 
        _tag = tag,
        _visualize=True,
        _outputs=[x_tile_node, y_tile_node],
        body=[x_load, y_load]
    )


### Node 4: create mma node
def create_mma(tag: SymCoord) -> cnc_ast.AstExtension:
    x_tile_inp = ast.Name(id="x_tile", ctx=ast.Load())
    y_tile_inp = ast.Name(id="y_tile", ctx=ast.Load())
    acc_inp = ast.Name(id="acc", ctx=ast.Load())
    acc_out = ast.Name(id="acc", ctx=ast.Store())
    addmm_node = statement_from_string("{acc_out} = cnc.addmm({acc_inp}, {x_tile_inp}, {y_tile_inp})", acc_out=acc_out, acc_inp=acc_inp, x_tile_inp=x_tile_inp, y_tile_inp=y_tile_inp)

    return cnc_ast.create(
        ast.Module, 
        _tag = tag,
        _visualize=True,
        _inputs=[acc_inp, x_tile_inp, y_tile_inp],
        _outputs=[acc_out],
        body=[addmm_node]
    )



### Node 5: create store node
def create_store_node(tag: SymCoord) -> cnc_ast.AstExtension:
    acc_inp = ast.Name(id="acc", ctx=ast.Load())

    store_sub = tag[0]
    store_node = statement_from_string("cnc.store({acc_inp}, out[{store_sub}])", acc_inp=acc_inp, store_sub=symcoord_to_ast(store_sub))

    return cnc_ast.create(
        ast.Module, 
        _tag = tag,
        _visualize=True,
        _inputs=[acc_inp],
        _outputs=[],
        body=[store_node]
    )


base_tag = (None, None)
grid_loop_node = create_grid_loop_node(base_tag)

grid_loop_node.visualize(filename="matmul", view=False)


ast.fix_missing_locations(grid_loop_node)
code = ast.unparse(grid_loop_node)
print("code: \n", code)




# for tile_m, tile_n in cnc.tile(out.shape): # This is a grid loop -- not executed as a loop on the GPU. But we will still genereate it. 
#     tag = (tile_m, tile_n, None)
#     acc = cute.zeros_like(out.tile, dtype=torch.float32)
#     for tile_k in cnc.tile(a, mode=[1]):
#         tag = tag + (tile_k, )

#         x_tile = cnc.load(x[tag[0], tag[2]])
#         y_tile = cnc.load(y[tag[2], tag[1]])

#         acc = cute.addmm(acc, x_tile, y_tile)
#     cnc.store(acc, out[tag[0], tag[1]])
