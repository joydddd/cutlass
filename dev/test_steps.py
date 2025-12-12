from cutlass.cnc_dsl.steps import step
import torch
import cutlass.cute as cute
import cutlass
from cutlass.cute.runtime import from_dlpack
from cutlass.cnc_dsl.trace import Context


# Helper functions to use tv_layout in the kernel
@cute.jit
def get_subtile_view(
    T: cute.Tensor,
    tv_layout: cute.Layout,
    tile_coord: cute.Coord,
    subtile_coord: cute.Coord,
) -> cute.Layout:
    """
    CuTe Kernel Helper function to get a view of subtile in a tensor.

    Args:
        T: tensor to get a view from
        tv_layout: specifies the tile/subtile partitioning
        tile_coord: the coordinate of the tile in the tensor
        subtile_coord: the coordinate of the subtile in the tile

    Returns:
        a view of the subtile at tile_coord, subtile_coord in T.
        The view can be used to load/store data from/to the subtile.
    """

    tile = T[(None, tile_coord)]
    tile = cute.composition(tile, tv_layout)
    subtile = cute.flatten(tile[(subtile_coord, None)])
    return subtile


@step(tag="coord")
def load_a_tile(
    gA: cute.Tensor, tv_layout: cute.Layout, coord: cute.Coord
) -> cute.Tensor:
    """
    Load a A tile from global memory into register.
    """
    bid, tid = coord

    gA_subtile = get_subtile_view(gA, tv_layout, tile_coord=bid, subtile_coord=tid)
    reg = gA_subtile.load()
    return reg


@step(tag="tag")
def add_constant(reg: cute.Tensor, tag: cute.Coord) -> cute.Tensor:
    """
    Add constant to each element in the register.
    """
    reg = reg + cutlass.BFloat16(2)
    return reg


@step(tag="coord", jit=cute.jit)
def store_a_tile(
    reg: cute.Tensor, gO: cute.Tensor, tv_layout: cute.Layout, coord: cute.Coord
) -> None:
    """
    Store a tile from register to global memory.
    """
    bid, tid = coord
    gO_subtile = get_subtile_view(gO, tv_layout, tile_coord=bid, subtile_coord=tid)
    gO_subtile.store(reg)


# CuTe kernel running on GPU
@cute.kernel
def add_kernel(gA: cute.Tensor, gO: cute.Tensor, tv_layout: cute.Layout):
    # Get the thread's position in the thread block. 2D, size = (M, N//8)
    tid_x, tid_y, _ = cute.arch.thread_idx()
    bid_x, _, _ = cute.arch.block_idx()

    for i in cutlass.range(gA.shape[1][1]):
        tag = ((bid_x, i), (tid_x, tid_y))

        # ---- ✨ Your Code Here ✨----
        # Load a A subtile from global memory into register
        reg = load_a_tile(gA, tv_layout, tag)
        # ---- ✨ End Code ✨----

        # Add constant to each element in the kernel
        reg = add_constant(reg, tag)

        # ---- ✨ Your Code Here ✨----
        # Store computed tile from register to corresponding global memory for O
        store_a_tile(reg, gO, tv_layout, tag)
        # ---- ✨ End Code ✨----


# Host function that launches the CuTe kernel
@cute.jit
def cute_add(A: cute.Tensor, OUT: cute.Tensor):
    # ---- ✨ Your Code Here ✨----
    # Compute tv_tiler and tv_layout for A & O.

    block_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    thr_layout = cute.make_ordered_layout((16, 8), order=(1, 0))
    tv_tiler, tv_layout = cute.make_layout_tv(block_layout, thr_layout)

    # ---- ✨ End Code ✨----

    # interpret A & O as our two-level layout.
    gA = cute.zipped_divide(A, tiler=tv_tiler)
    gO = cute.zipped_divide(OUT, tiler=tv_tiler)

    # Launch CuTe kernel
    add_kernel(gA, gO, tv_layout).launch(
        grid=[gA.shape[1][0], 1, 1],
        block=[tv_layout.shape[0][0], tv_layout.shape[0][1], 1],
    )


def jit_cute_add(A: torch.Tensor) -> torch.Tensor:
    torchO = torch.empty_like(A)
    cuteA = from_dlpack(A, assumed_align=16)
    cuteO = from_dlpack(torchO, assumed_align=16)

    with Context():
        cute.compile(cute_add, cuteA, cuteO)

        ctx = Context.current()
        assert ctx is not None

        print("Step Registry: ", ctx.step_registry)
        print("Step Origin Registry: ", ctx.step_origin_registry)

    # breakpoint()

    # compiled_func(cuteA, cuteO)
    return torchO


M = 8192
N = 16384
A = torch.rand(M, N, dtype=torch.bfloat16, device="cuda")


jit_cute_add(A)
