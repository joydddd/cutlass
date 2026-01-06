import dataclasses
from typing import Tuple, TYPE_CHECKING

from .cnc_graph import CnCNode

from .. import ast as cnc_ast

if TYPE_CHECKING:
    from ..steps import Step


@dataclasses.dataclass
class StepState(CnCNode):
    name: str
    origin: "Step"
    inputs: list[cnc_ast.AstExtension]
    outputs: list[cnc_ast.AstExtension]

    def __repr__(self):
        return f"{self.origin.__repr__()} [{self.tag}]"

    def dump(self) -> Tuple[str, ...]:
        return (f"{self.__repr__()} id={self.id}",)
