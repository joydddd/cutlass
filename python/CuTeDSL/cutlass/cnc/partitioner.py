from __future__ import annotations
from cutlass.cute import *


from cutlass.cutlass_dsl import (
    dsl_user_op,
)


class Partitioner:
    tiler: Layout
    _tv_layout: Layout | None = None

    def __init__(self, tiler_mn: Layout, tv_layout: Layout | None = None) -> None:
        self._tiler_mn = tiler_mn
        self._tv_layout = tv_layout
    

    def __str__(self) -> str:
        return f"Partitioner(tiler={self._tiler_mn}, tv_layout={self._tv_layout})"
    
    @property
    def tv_layout(self) -> Layout:
        return self._tv_layout

    @property
    def tiler_mn(self) -> Layout:
        return self._tiler_mn
    
    @dsl_user_op
    def tiler(self, *, loc=None, ip=None) -> Layout:
        return product_each(self._tiler_mn.shape, loc=loc, ip=ip)
    

    @dsl_user_op
    def apply_tiler(self, tensor: Tensor, *, loc=None, ip=None) -> Tensor:
        return zipped_divide(tensor, self._tiler_mn, loc=loc, ip=ip)
    
    @dsl_user_op
    def get_tile(self, tensor: Tensor, tag: Coord, *, loc=None, ip=None) -> Tensor:
        tiled_tensor = self.apply_tiler(tensor, loc=loc, ip=ip)
        tile = tiled_tensor[(tag, None)]
        frag_tile = composition(tile, self.tv_layout, loc=loc, ip=ip)
        return frag_tile
    
    

    


    @dsl_user_op
    @staticmethod
    def from_tv(thr_layout: Layout, val_layout: Layout, *, loc=None, ip=None) -> Partitioner:
        """Create a TV partitioner for a given thread and value layout."""
        tiler_mn = raked_product(thr_layout, val_layout, loc=loc, ip=ip)
        _, tv_layout = make_layout_tv(thr_layout, val_layout, loc=loc, ip=ip)
        return Partitioner(tiler_mn, tv_layout)
    

    @dsl_user_op
    @staticmethod
    def from_cotile(atom_layout_tv: Layout, data_layout: Layout, *, loc=None, ip=None) -> Partitioner:
        """atom_layout_tv : (tid, vid) -> data addr
            data_layout : data coord -> data addr
        """
        assert is_static(atom_layout_tv.type) and is_static(data_layout.type), (
            "atom_layout_tv and data_layout must be static"
        )

        # data addr -> data coord
        inv_layout_ = left_inverse(data_layout, loc=loc, ip=ip)
        inv_data_layout = make_layout(
            (inv_layout_.shape, (1)), stride=(inv_layout_.stride, (0)), loc=loc, ip=ip
        )

        # (tid,vid) -> data_coord
        layout_tv_data = composition(inv_data_layout, atom_layout_tv, loc=loc, ip=ip)


        # check validity
        atom_layout_v_to_check = coalesce(
            make_layout(
                atom_layout_tv.shape[1], stride=atom_layout_tv.stride[1], loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
        data_layout_v_to_check = coalesce(
            composition(
                data_layout,
                make_layout(
                    layout_tv_data.shape[1], stride=layout_tv_data.stride[1], loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        )
        assert data_layout_v_to_check == atom_layout_v_to_check, (
            "the memory pointed to by atom_layout_tv does not exist in the data_layout."
        )


        flat_data_shape = product_each(data_layout.shape, loc=loc, ip=ip)
        tiler = tuple(
            filter(
                composition(
                    make_layout(
                        flat_data_shape,
                        stride=tuple(
                            0 if j != i else 1 for j in range(rank(flat_data_shape))
                        ),
                        loc=loc,
                        ip=ip,
                    ),
                    layout_tv_data,
                    loc=loc,
                    ip=ip,
                ),
                loc=loc,
                ip=ip,
            )
            for i in range(rank(flat_data_shape))
        )

        print(tiler)
        # tile_coord -> data_coord
        tile2data = composition(
            make_layout(flat_data_shape, loc=loc, ip=ip), tiler, loc=loc, ip=ip
        )
        # (tid,vid) -> tile_coord
        layout_tv = composition(
            left_inverse(tile2data, loc=loc, ip=ip), layout_tv_data, loc=loc, ip=ip
        )

        return Partitioner(data_layout, layout_tv)



make_tv_partitioner = Partitioner.from_tv
make_partitioner = Partitioner
make_cotiled_partitioner = Partitioner.from_cotile