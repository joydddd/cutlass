import ast


import cutlass.cnc.ast as cnc_ast


### Node 0: The grid loop. 
def create_grid_loop_node(tag: ast.AST) -> cnc_ast.AstExtention:
    iter_node = ast.parse("cnc.tile(out.shape)", mode="eval").body


    tag_node = ast.Name(id="tag1", ctx=ast.Load())

    body_nodes = [
        create_init_node(tag_node),
        create_inner_loop_node(tag_node),
        create_store_node(tag_node)
    ]

    return cnc_ast.create(
        ast.For,  
        _visualize=True,
        _tag = tag, 
        target=tag_node,
        iter=iter_node,
        body=body_nodes, 
        orelse=[],
        type_comment=None
    )



### Node 1: The init node. 
def create_init_node(tag: ast.AST) -> cnc_ast.AstExtention:

    compute_node = ast.parse("cnc.zeros_like(out.tile, dtype=torch.float32)", mode="eval").body

    output = ast.Name(id="acc", ctx=ast.Load())

    return cnc_ast.create(
        ast.Assign, 
        _tag = tag,
        _visualize=True,
        targets=[output],
        value=compute_node,
        type_comment=None,
    )

### Node 2: The inner loop node, 
def create_inner_loop_node(tag: ast.AST) -> cnc_ast.AstExtention:
    iter_node = ast.parse(f"cnc.tile(a, mode=[1])", mode="eval").body
    target = ast.Name(id="tag2", ctx=ast.Load())

    return cnc_ast.create(
        ast.For,  
        _visualize=True,
        _tag = tag,
        target=target,
        iter=iter_node,
        body=[
           create_load_xy(target),
           create_mma(target), 
        ], 
        orelse=[],
        type_comment=None
    )


### Node 3: create load node
def create_load_xy(tag: ast.AST) -> cnc_ast.AstExtention:
    compute_node = ast.parse(f"cnc.load(x[tag[0], tag[2]]), cnc.load(y[tag[2], tag[1]])", mode="eval").body
    output = ast.parse(f"tile_x, tile_y", mode="eval").body

    return cnc_ast.create(
        ast.Assign, 
        _tag = tag,
        _visualize=True,
        targets=[output],
        value=compute_node,
        type_comment=None,
    )



### Node 4: create mma node
def create_mma(tag: ast.AST) -> cnc_ast.AstExtention:
    compute_node = ast.parse(f"cnc.addmm(acc, tile_x, tile_y)", mode="eval").body
    output = ast.Name(id="acc", ctx=ast.Load())

    return cnc_ast.create(
        ast.Assign, 
        _tag = tag,
        _visualize=True,
        targets=[output],
        value=compute_node,
        type_comment=None,
    )



### Node 5: create store node
def create_store_node(tag: ast.AST) -> cnc_ast.AstExtention:
    store_node = ast.parse(f"cnc.store(acc, out[tag[0], tag[1]])", mode="eval").body

    store_fields = {field: getattr(store_node, field) for field in store_node._fields}

    return cnc_ast.create(
        ast.Call, 
        _tag = tag,
        _visualize=True,
        **store_fields, 
    )




tag_node = ast.Name(id="tag", ctx=ast.Load())
grid_loop_node = create_grid_loop_node(tag_node)

grid_loop_node.visualize(filename="matmul", view=False)




# for tile_m, tile_n in hl.tile(out.shape): # This is a grid loop -- not executed as a loop on the GPU. But we will still genereate it. 
#     tag = cnc_ast.CnCTag((tile_m, tile_n, None))
#     acc = cute.zeros_like(out.tile, dtype=torch.float32)
#     for tile_k in hl.tile(a, mode=[1]):
#         tag = tag + (tile_k, )

#         x_tile = cnc.load(x[tag[0], tag[2]])
#         y_tile = cnc.load(y[tag[2], tag[1]])

#         acc = torch.addmm(acc, x_tile, y_tile)
#     cnc.store(acc, out[tag[0], tag[1]])
