from dataclasses import dataclass
import ast
import typing
import cutlass.cute as cute



class SourceLocation:
    def __init__(self, file_name: str, line_number: int, column_number: int) -> None:
        self.file_name = file_name
        self.line_number = line_number
        self.column_number = column_number



def _ast_to_graph(
    node: ast.AST,
    dot=None,
    parent_id: str | None = None,
    counter: list[int] | None = None,
    visible_ids: list[str] | None = None,
    edge_label: str | None = None,
):
    """
    Build a Graphviz graph for the AST, but only emitting nodes that are:
      - instances of AstExtention, and
      - have _visualize set to True.

    We still traverse all children so that visible nodes stay connected even
    if there are non-visual (or non-extended) nodes in between.
    """
    from graphviz import Digraph  # pip install graphviz

    if dot is None:
        dot = Digraph()
        # Make the dataflow-style chains read left-to-right and align nicely,
        # and use rounded, filled nodes similar to the reference sketch.
        dot.attr(rankdir="LR", splines="spline", nodesep="0.8", ranksep="0.8")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="#a5d8ff")
        dot.attr("edge", arrowsize="0.7")
    if counter is None:
        counter = [0]
    if visible_ids is None:
        visible_ids = []

    # Decide whether this node should appear in the visualization.
    is_extended = isinstance(node, AstExtention)
    should_draw = is_extended and getattr(node, "_visualize", False)

    my_id = parent_id

    if should_draw:
        my_id = str(counter[0])
        counter[0] += 1
        visible_ids.append(my_id)

        # Basic label: class name, plus some extra info for a few node types
        label = type(node).__name__
        if isinstance(node, ast.Assign):
            # For Assign nodes, show the RHS expression instead of "Assign".
            try:
                value_str = ast.unparse(node.value)
            except Exception:
                value_str = ast.dump(node.value)
            label = value_str.replace("\n", " ")
        elif isinstance(node, ast.For):
            # For-loops: keep the label short and friendly.
            label = "for"
        elif isinstance(node, ast.Name):
            label += f"\\nid={node.id}"

        # Render _tag as a human-readable string if present.
        tag = getattr(node, "_tag", None)
        if tag is not None:
            if isinstance(tag, ast.AST):
                try:
                    tag_str = ast.unparse(tag)
                except Exception:
                    tag_str = ast.dump(tag)
            else:
                tag_str = str(tag)
            # Keep label on one line for Graphviz; escape newlines.
            tag_str = tag_str.replace("\n", " ")
            label += f"\\ntag={tag_str}"

        node_shape = "diamond" if isinstance(node, ast.For) else "box"
        dot.node(my_id, label=label, shape=node_shape, fontsize="10")

        if parent_id is not None:
            if edge_label is not None:
                dot.edge(parent_id, my_id, label=edge_label)
            else:
                dot.edge(parent_id, my_id)

    # Decide what ID to propagate to children (nearest drawn ancestor).
    current_parent = my_id

    # Special handling for For-loops: visualize body as a dataflow-style chain.
    if isinstance(node, ast.For):
        # First, recurse into non-body children in the usual tree style.
        for field_name, value in ast.iter_fields(node):
            if field_name == "body":
                continue
            if isinstance(value, ast.AST):
                _ast_to_graph(value, dot, current_parent, counter, visible_ids)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        _ast_to_graph(item, dot, current_parent, counter, visible_ids)

        # Helper to stringify a node's assignment/loop target(s).
        def _target_label(owner: ast.AST) -> str | None:
            # Single target (For/AsyncFor or custom nodes).
            if hasattr(owner, "target"):
                t = getattr(owner, "target")
                if isinstance(t, ast.AST):
                    try:
                        return ast.unparse(t).replace("\n", " ")
                    except Exception:
                        return ast.dump(t)
            # Multiple targets (Assign, AugAssign-style custom nodes).
            if hasattr(owner, "targets"):
                ts = getattr(owner, "targets", [])
                parts: list[str] = []
                for t in ts:
                    if isinstance(t, ast.AST):
                        try:
                            parts.append(ast.unparse(t).replace("\n", " "))
                        except Exception:
                            parts.append(ast.dump(t))
                if parts:
                    return ", ".join(parts)
            return None

        # Then, chain body statements: parent -> body[0] -> body[1] -> ...
        prev_id = current_parent
        last_body_id: str | None = None
        # Track which node's target(s) should label the next edge in the chain.
        label_owner: ast.AST | None = node

        for stmt in node.body:
            label_for_edge = _target_label(label_owner) if label_owner is not None else None

            _, child_id, _ = _ast_to_graph(
                stmt,
                dot,
                prev_id,
                counter,
                visible_ids,
                edge_label=label_for_edge,
            )
            # Only advance the chain if this statement (or one of its children)
            # produced a visual node.
            if child_id is not None:
                prev_id = child_id
                last_body_id = child_id
                # For non-loop statements, the next edge in the chain should be
                # labeled by this statement's own target/targets, if any. For
                # nested loops, we keep the previous label owner so that the
                # edge from the nested loop's last body node to the next
                # statement is labeled by the producing assignment (e.g. "acc"),
                # not by the inner loop's induction variable.
                if not isinstance(stmt, ast.For):
                    label_owner = stmt

        # Add a loop-back edge from the last body node to the loop node itself,
        # to visually indicate the iteration. Mark this edge as
        # constraint="false" so it doesn't affect the left-to-right ordering
        # of the main dataflow chain.
        if should_draw and last_body_id is not None and my_id is not None and last_body_id != my_id:
            dot.edge(last_body_id, my_id, constraint="false")

        # For chaining at the parent level, we want the "last" node of this
        # loop to be the last visible statement in its body, so that nested
        # loops are flattened in the main dataflow chain:
        #   for0 -> body0 -> for1 -> body1_0 -> body1_1 -> body2 -> ...
        effective_last_id = last_body_id if last_body_id is not None else (my_id if should_draw else parent_id)
    else:
        # Default: regular AST tree edges.
        for child in ast.iter_child_nodes(node):
            _ast_to_graph(child, dot, current_parent, counter, visible_ids)

    # Return both the graph and the "last drawn" node id for chaining, plus the
    # list of all visible node ids so the caller can align them.
    last_id = effective_last_id if isinstance(node, ast.For) else (my_id if should_draw else parent_id)
    return dot, last_id, visible_ids


class AstExtention: 
    _fields: tuple[str, ...]
    

    def __init__(
        self, 
        *,
        _epilogue_location: SourceLocation | None = None, 
        _prologue_location: SourceLocation | None = None, 
        _tag: ast.AST,
        _visualize: bool = False, 
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._epilogue_location: SourceLocation = _epilogue_location
        self._prologue_location: SourceLocation = _prologue_location
        self._tag: ast.AST = _tag
        self._visualize: bool = _visualize
    

    def visualize(self, filename: str = "cnc_ast", view: bool = True) -> None:
        """
        Render this extended AST node as a Graphviz graph.

        Requires:
            pip install graphviz
        and the Graphviz 'dot' binary installed on your system.
        """
        dot, _, visible_ids = _ast_to_graph(self)

        # Force all visible nodes to share the same rank so they align
        # horizontally, regardless of shape differences.
        if visible_ids:
            with dot.subgraph() as s:
                s.attr(rank="same")
                for vid in visible_ids:
                    s.node(vid)

        dot.render(filename, format="png", view=view)


_to_extended: dict[type[ast.AST], type[ast.AST]] = {}

def get_wrapper_cls(cls: type[ast.AST]) -> type[ast.AST]:
    if new_cls := _to_extended.get(cls):
        return new_cls

    class Wrapper(AstExtention, cls):
        pass

    Wrapper.__name__ = cls.__name__
    rv = typing.cast("type[ast.AST]", Wrapper)
    _to_extended[cls] = rv
    return rv

def create(cls, **fields: object):
    result = get_wrapper_cls(cls)(**fields)
    assert isinstance(result, AstExtention)
    return result
        