from __future__ import annotations
import dataclasses
from typing import ClassVar, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from .step import StepState
    from .loop import LoopState
    from .kernel import KernelBuilder


from ...cnc.symint import SymCoord, SymTile, SymIntSource
from .. import ast as cnc_ast


@dataclasses.dataclass
class CnCNode:
    tag: SymCoord | None
    id: int | None
    ast: cnc_ast.AstExtension | None
    parent: CnCNode | None

    def dump(self) -> Tuple[str, ...]:
        raise NotImplementedError("Subclass must implement dump method.")


class CnCContext:
    _current: ClassVar["CnCContext | None"] = None

    node_registry: dict[int, CnCNode] = {}  # node_id -> CnCNode
    _next_node_id: int = 0  # node id shared by loops and steps.

    kernel_registry: dict[int, KernelBuilder] = {}  # kernel_id -> KernelBuilder
    _next_kernel_id: int = 0  # next kernel id to register

    _current_kernel_id: int | None = None

    symtile_source: SymIntSource = SymIntSource(prefix="tile")
    symcoord_source: SymIntSource = SymIntSource(prefix="coord")

    def __init__(self):
        super().__init__()

    def __enter__(self) -> "CnCContext":
        # Save previous context to support nesting
        self._prev_context = CnCContext._current
        CnCContext._current = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Restore previous context when leaving the with-block
        CnCContext._current = getattr(self, "_prev_context", None)

    @staticmethod
    def current() -> "CnCContext | None":
        """
        Return the current active CnCContext, or None if no CnCContext is active.

        Intended usage:

            with CnCContext():
                ctx = CnCContext.current()
                assert ctx is not None
        """
        return CnCContext._current

    @property
    def current_kernel(self) -> KernelBuilder | None:
        if self._current_kernel_id is None:
            return None

        return self.kernel_registry[self._current_kernel_id]

    def print(self) -> None:
        for kid, kernel in self.kernel_registry.items():
            print(f"Kernel {kid}:")
            print(kernel.dump())
        # for sid, step in self.node_registry.items():
        #     print(f"Node {sid}:")
        #     print(step)

    ################### Get new symbolic tag or tile #################
    def create_new_coord(self) -> SymCoord:
        return self.symcoord_source.create_new()

    def create_new_tile(self) -> SymTile:
        return self.symtile_source.create_new()

    ################ Register Steps, Loops and Kernels #################

    def register_node(self, node: CnCNode) -> None:
        if node.id is not None:
            raise RuntimeError(
                f"Node id is already set for {node}. Duplicated node registration."
            )
        node.id = self._next_node_id
        self._next_node_id += 1

        # check duplicated node
        for existing in self.node_registry.values():
            if existing is node:
                raise ValueError(
                    f"Node {node} already registered in node registry as ID={existing.id}. "
                )
        self.node_registry[node.id] = node

    def register_step(self, step: StepState) -> None:
        self.register_node(step)

        # Add step to the current kernel.
        if self.current_kernel is not None:
            self.current_kernel.add_step(step)

    def register_loop(self, loop: LoopState) -> None:
        self.register_node(loop)

        # Add loop to the current kernel.
        if self.current_kernel is not None:
            self.current_kernel.add_loop(loop)

    def register_kernel(self, kernel_builder: KernelBuilder) -> None:
        # Check if already in a kernel scope.
        if self.current_kernel is not None:
            raise ValueError(
                "Already in a kernel. Can't launch another kernel within a kernel."
            )

        if kernel_builder.id is not None:
            raise ValueError(
                f"Kernel builder id is already set for {kernel_builder}. Duplicated kernel registration."
            )

        kernel_builder.id = self._next_kernel_id
        self._next_kernel_id += 1

        self.kernel_registry[kernel_builder.id] = kernel_builder
        self.register_loop(
            kernel_builder.grid_loop
        )  # Register the grid loop for the kernel.
        self._current_kernel_id = kernel_builder.id

    def exit_kernel(self) -> None:
        if self._current_kernel_id is None:
            raise ValueError(
                "Not in a kernel. Can't exit a kernel that is not launched."
            )
        self._current_kernel_id = None
