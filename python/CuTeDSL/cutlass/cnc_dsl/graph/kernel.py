from __future__ import annotations
import dataclasses

from .step import StepState
from .loop import LoopState, GridLoopState

from ...base_dsl.dsl import BaseDSL, DSLRuntimeError
from .. import ast as cnc_ast

from typing import Callable


@dataclasses.dataclass
class KernelBuilder:
    id: int | None
    name: str
    inputs: list[cnc_ast.AstExtension]
    outputs: list[cnc_ast.AstExtension]

    _grid_loop: GridLoopState
    _current_loop: LoopState | None

    in_kernel_scope: bool = True

    def __repr__(self):
        return f"[[[Kernel]]] {self.name} (id={self.id})"

    @staticmethod
    def create_from_kernel_launch(
        kernel_name: str,
        config: BaseDSL.LaunchConfig,
        funcBody: Callable,
        *args,
        **kwargs,
    ) -> "KernelBuilder":
        grid_loop = GridLoopState.create_from_launch_config(config)
        return KernelBuilder(
            id=None,
            name=kernel_name,
            inputs=[],
            outputs=[],
            _grid_loop=grid_loop,
            _current_loop=grid_loop,
        )

    @property
    def grid_loop(self) -> GridLoopState:
        return self._grid_loop

    def dump(self) -> str:
        dump_str = self.__repr__()
        for node_dump in self._grid_loop.dump():
            dump_str += f"\n{node_dump}"
        return dump_str

    def add_step(self, step: StepState) -> None:
        if not self.in_kernel_scope:
            raise DSLRuntimeError("Can't add step outside of kernel scope.")
        self._current_loop.add_step(step)

    def add_loop(self, loop: LoopState) -> None:
        if not self.in_kernel_scope:
            raise DSLRuntimeError("Can't add loop outside of kernel scope.")
        self._current_loop.add_loop(loop)
        self._current_loop = loop

    def exit_loop(self) -> None:
        if self._current_loop is self._grid_loop:
            raise DSLRuntimeError(
                "Can't exit the grid loop -- that means we are exiting the kernel."
            )
        if self._current_loop.parent is None:
            raise DSLRuntimeError("Current loop has no parent. ")
        self._current_loop = self._current_loop.parent

    def exit_kernel(self) -> None:
        if not self.in_kernel_scope:
            raise DSLRuntimeError("Can't exit kernel -- not in kernel scope.")
        if self._current_loop is not self._grid_loop:
            raise DSLRuntimeError("Cannot exit kernel -- not in the grid loop. ")

        self.in_kernel_scope = False
        self._current_loop = None
