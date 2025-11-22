import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

M = 8192
K = 8192
N = 4096
A = torch.rand(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.rand(N, K, dtype=torch.bfloat16, device="cuda").transpose(0, 1)



# Helper functions to use tv_layout in the kernel
@cute.jit
def get_subtile_view(T: cute.Tensor, tv_layout: cute.Layout, tile_coord: cute.Coord, subtile_coord:cute.Coord) -> cute.Layout:
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
    print("tile = ", tile)
    tile = cute.composition(tile, tv_layout)
    print("tile after composition = ", tile)
    subtile = cute.flatten(tile[(subtile_coord, None)])
    print("subtile = ", subtile)
    return subtile

@cute.jit
def prepare_smem_buffer(gA: cute.Tensor, gB: cute.Tensor) -> tuple[cute.Tensor, cute.Tensor]:
    """
    Allocate shared memory buffer for caching A & B tiles.  
    """
    smem = cutlass.utils.SmemAllocator()
    A_layout = cute.make_ordered_layout(gA.shape[0], order=(1, 0))
    B_layout = cute.make_ordered_layout(gB.shape[0], order=(1, 0))

    # ---- ✨ Your Code Here ✨----
    # Allocate shared memory buffer for A & B tiles. 
    sA_tile = smem.allocate_tensor(element_type=gA.element_type, layout=A_layout, byte_alignment=16)
    sB_tile = smem.allocate_tensor(element_type=gB.element_type, layout=B_layout, byte_alignment=16)
    # ---- ✨ End Code ✨----

    return sA_tile, sB_tile

@cute.jit
def prepare_rmem_buffer(C_subtile:cute.Tensor, dtype) -> cute.Tensor:
    """
    Allocate a register accumulator for Csubtile, and initialize it to zeros.
    """
    # acc_rmem = cute.make_rmem_tensor_like(subtileC, ctype=cute.Float32) For version >= 4.3.0
    rACC = cute.make_fragment_like(C_subtile, dtype=dtype) # Use float32 accumulation for increated accuracy. 
    rACC.fill(cute.Float32(0.0))
    return rACC

@cute.jit 
def load_tile_g2s(gT:cute.Tensor, sT_tile:cute.Tensor, tv_layout:cute.Layout, tile_coord:cute.Coord):
    """
    Load a tile from global memory to shared memory. 
    """
    cute.arch.sync_threads()

    tid_x, tid_y, _ = cute.arch.thread_idx()
    subtile_coord = (tid_x, tid_y)
    
    # ---- ✨ Your Code Here ✨----
    # Load a subtile from global memory. 
    
    gT_subtile = get_subtile_view(gT, tv_layout, tile_coord = tile_coord, subtile_coord = subtile_coord)
    reg = gT_subtile.load()
    # ---- ✨ End Code ✨----

    sT_tv_partitioned = cute.composition(sT_tile, tv_layout)
    sT_subtile = cute.flatten(sT_tv_partitioned[(subtile_coord, None)])

    # ---- ✨ Your Code Here ✨----
    # Store the subtile to shared memory.
    sT_subtile.store(reg)
    # ---- ✨ End Code ✨----

    cute.arch.sync_threads()
    return sT_tile

@cute.jit
def subtile_mmaT_s2r(sA_tile: cute.Tensor, sBT_tile: cute.Tensor, rC_subtile: cute.Tensor, tv_layout:cute.Layout, subtile_coord: cute.Coord) -> cute.Tensor:
    cute.arch.sync_threads()
    n_offset = tv_layout.shape[1][0] * subtile_coord[0]
    m_offset = tv_layout.shape[1][1] * subtile_coord[1]

    # Itereate each element in C subtile. 
    for i in cutlass.range_constexpr(rC_subtile.shape[0]):
        for j in cutlass.range_constexpr(rC_subtile.shape[1]): 
            m = m_offset + j
            n = n_offset + i
            a_row = sA_tile[(m, None)].load().to(cutlass.Float32)
            b_col = sBT_tile[(n, None)].load().to(cutlass.Float32)
            c_vec = a_row * b_col
            c = c_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)
            rC_subtile[(i, j)] = c + rC_subtile[(i, j)]
    cute.arch.sync_threads()
        

@cute.kernel
def mamtul_kernel(gA: cute.Tensor, gBT: cute.Tensor, gC: cute.Tensor, AB_tv_layout: cute.Layout, C_tv_layout: cute.Layout):
    tid_x, tid_y, _ = cute.arch.thread_idx()
    bid_x, bid_y, _ = cute.arch.block_idx()

    # prepare shared memory buffer for A & B tiles. 
    sA_tile, sBT_tile = prepare_smem_buffer(gA, gBT)


    # prologue: create Csubtile accumulator, and initialize it to zeros. 
    gC_subtile = get_subtile_view(gC, C_tv_layout, tile_coord = (bid_x, bid_y), subtile_coord = (tid_x, tid_y))
    rC_subtile = prepare_rmem_buffer(gC_subtile, cute.Float32)

    breakpoint()

    for k_tile_coord in cutlass.range(gA.shape[1][1], unroll=1):
        breakpoint()
        # Step 1: Load A & B tiles from global memory to shared memory. 
        load_tile_g2s(gA, sA_tile, AB_tv_layout, (bid_x, k_tile_coord))
        load_tile_g2s(gBT, sBT_tile, AB_tv_layout, (bid_y, k_tile_coord))
        # cute.printf("tile_coord %d %d\n", bid_x, k_tile_coord)


        # Step 2: iteratively compute one element in C subtile, and store that to rC. 
        subtile_mmaT_s2r(sA_tile, sBT_tile, rC_subtile, C_tv_layout, (tid_x, tid_y))
    
    gC_subtile.store(rC_subtile.load().to(gC.element_type))

    
# Host function that launches the CuTe kernel
# @cute.jit
def cute_matmul(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):

    BT = cute.composition(B, cute.make_ordered_layout(B.shape, order=(1, 0)))

    # ---- ✨ Your Code Here ✨----
    # Generate 2-level tv_layout for AB & C: AB_tiler, AB_layout, C_tiler, C_layout
    block_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    subtile_layout_AB = cute.make_ordered_layout((16, 8), order=(1, 0))
    subtile_layout_C = cute.make_ordered_layout((16, 2), order=(1, 0))
    AB_tiler, AB_tv_layout = cute.make_layout_tv(block_layout, subtile_layout_AB)
    C_tiler, C_tv_layout = cute.make_layout_tv(block_layout, subtile_layout_C)
    # ---- ✨ End Code ✨----

    # partition A & B & C according to our two-level layout. 
    gA = cute.zipped_divide(A, tiler=AB_tiler)
    gBT = cute.zipped_divide(BT, tiler=AB_tiler)
    gC = cute.zipped_divide(C, tiler=C_tiler)

    # Display AB & C layouts. 
    # display_tv_layout(AB_layout, AB_tiler)
    # display_tv_layout(C_layout, C_tiler)


    # Launch CuTe kernel 
    mamtul_kernel(gA, gBT, gC, AB_tv_layout, C_tv_layout).launch(
        grid=[gC.shape[1][0], gC.shape[1][1], 1], 
        block=[C_tv_layout.shape[0][0], C_tv_layout.shape[0][1], 1],
    )

def jit_cute_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), dtype=A.dtype, device=A.device)

    cuteA = from_dlpack(A, assumed_align=16)
    cuteBT = from_dlpack(B.transpose(0, 1), assumed_align=16)
    cuteC = from_dlpack(C, assumed_align=16)

    cute_matmul(cuteA, cuteBT, cuteC)

    return C



def cute_compile_n_run_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), dtype=A.dtype, device=A.device)

    cuteA = from_dlpack(A, assumed_align=16)
    cuteA = cuteA.mark_layout_dynamic(leading_dim=1)
    cuteB = from_dlpack(B, assumed_align=16)
    cuteB = cuteB.mark_layout_dynamic(leading_dim=0)
    cuteC = from_dlpack(C, assumed_align=16)


    cute_matmul(cuteA, cuteB, cuteC)

    # compiled_func = cute.compile(cute_matmul, cuteA, cuteBT, cuteC)

    # compiled_func(cuteA, cuteBT, cuteC)

C = cute_compile_n_run_matmul(A, B)
print(C)