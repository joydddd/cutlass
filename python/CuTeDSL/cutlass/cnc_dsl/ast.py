import ast
from ..cnc.symint import SymCoord


class SourceLocation:
    def __init__(self, file_name: str, line_number: int, column_number: int) -> None:
        self.file_name = file_name
        self.line_number = line_number
        self.column_number = column_number


class AstExtension:
    def __init__(
        self,
        *,
        _epilogue_location: SourceLocation | None = None,
        _prologue_location: SourceLocation | None = None,
        _tag: SymCoord,
        _new_tag: SymCoord | None = None,
        _visualize: bool = False,
        _inputs: list[ast.AST] = [],
        _outputs: list[ast.AST] = [],
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._epilogue_location: SourceLocation = _epilogue_location
        self._prologue_location: SourceLocation = _prologue_location
        self._tag: SymCoord = _tag
        self._new_tag: SymCoord | None = _new_tag
        self._visualize: bool = _visualize
        self._inputs: list[ast.AST] = _inputs
        self._outputs: list[ast.AST] = _outputs

    # def visualize(self, filename: str = "cnc_ast", view: bool = True) -> None:
    #     """
    #     Render this extended AST node as a Graphviz graph.

    #     Requires:
    #         pip install graphviz
    #     and the Graphviz 'dot' binary installed on your system.
    #     """
    #     dot, _, visible_ids = _ast_to_graph(self)

    #     # Force all visible nodes to share the same rank so they align
    #     # horizontally, regardless of shape differences.
    #     if visible_ids:
    #         with dot.subgraph() as s:
    #             s.attr(rank="same")
    #             for vid in visible_ids:
    #                 s.node(vid)

    #     dot.render(filename, format="png", view=view)


# _to_extended: dict[type[ast.AST], type[ast.AST]] = {}


# def get_wrapper_cls(cls: type[ast.AST]) -> type[ast.AST]:
#     if new_cls := _to_extended.get(cls):
#         return new_cls

#     class Wrapper(AstExtension, cls):
#         pass

#     Wrapper.__name__ = cls.__name__
#     rv = typing.cast("type[ast.AST]", Wrapper)
#     _to_extended[cls] = rv
#     return rv


# def create(cls, **fields: object):
#     result = get_wrapper_cls(cls)(**fields)
#     assert isinstance(result, AstExtension)
#     return result
