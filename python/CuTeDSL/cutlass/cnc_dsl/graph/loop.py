from __future__ import annotations
import dataclasses
import ast
import sympy

from .step import StepState
from .cnc_graph import CnCNode, CnCContext


from ...cnc.symint import SymTile
from ...base_dsl.dsl import BaseDSL

from typing import Tuple


@dataclasses.dataclass
class LoopDimInfo:
    static_unroll: bool
    tile_dim_name: str | None
    tile_dim_expr: sympy.Expr | None


@dataclasses.dataclass
class LoopState(CnCNode):
    tile_id_to_info: dict[int, LoopDimInfo]
    inner_nodes: list[CnCNode]
    target: SymTile

    def dump(self) -> Tuple[str, ...]:
        dump_str: Tuple[str, ...] = (f"{self.__repr__()} id={self.id}",)

        inner_node_dumps: Tuple[str, ...] = ()
        for node in self.inner_nodes:
            inner_node_dumps += node.dump()

        for dump in inner_node_dumps:
            dump_str += (f"\t{dump}",)

        return dump_str

    def add_step(self, step: StepState) -> None:
        self.inner_nodes.append(step)
        step.parent = self

    def add_loop(self, loop: LoopState) -> None:
        self.inner_nodes.append(loop)
        loop.parent = self


@dataclasses.dataclass
class KernelLoopState(LoopState):
    def __repr__(self):
        return f"<KernelLoop> {self.target} [{self.tag}]"

    @staticmethod
    def create_from_ast(node: list[ast.AST]) -> "KernelLoopState":
        target = CnCContext.current().create_new_tile()
        return KernelLoopState(
            tag=None,
            id=None,
            ast=None,
            parent=None,
            tile_id_to_info={},
            inner_nodes=[],
            target=target,
        )

    @staticmethod
    def create_from_loop_selector(
        loopbody,
        start,
        stop,
        step,
        write_args,
        full_write_args_count,
        write_args_names,
        unroll,
        unroll_full,
        prefetch_stages,
    ) -> "KernelLoopState":
        target = CnCContext.current().create_new_tile()
        return KernelLoopState(
            tag=None,
            id=None,
            ast=None,
            parent=None,
            tile_id_to_info={},
            inner_nodes=[],
            target=target,
        )


@dataclasses.dataclass
class GridLoopState(LoopState):
    def __repr__(self):
        return f"<GridLoop> {self.target} [{self.tag}]"

    @staticmethod
    def create_from_launch_config(config: BaseDSL.LaunchConfig) -> "GridLoopState":
        target = CnCContext.current().create_new_tile()
        return GridLoopState(
            tag=None,
            id=None,
            ast=None,
            parent=None,
            tile_id_to_info={},
            inner_nodes=[],
            target=target,
        )
