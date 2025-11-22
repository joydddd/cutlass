from dataclasses import dataclass
import ast
import typing
import cutlass.cute as cute
from .symbolic_tag import SymCoord, SymInt



class SourceLocation:
    def __init__(self, file_name: str, line_number: int, column_number: int) -> None:
        self.file_name = file_name
        self.line_number = line_number
        self.column_number = column_number



class AstGraphVisitor(ast.NodeVisitor):
    """
    AST visitor that builds a Graphviz graph.
    
    Only visits nodes that are instances of AstExtension with _visualize=True.
    Only handles two node types: Module and For.
    """
    
    def __init__(self, dot=None):
        from graphviz import Digraph
        
        if dot is None:
            dot = Digraph()
            dot.attr(rankdir="LR", splines="spline", nodesep="0.8", ranksep="0.8")
            dot.attr("node", shape="plaintext")
            dot.attr("edge", arrowsize="0.7")
        
        self.dot = dot
        self.counter = 0
        self.visible_ids: list[str] = []
        self.parent_id: str | None = None
        self.edge_label: str | None = None
        self.last_id: str | None = None
    
    def should_process(self, node: ast.AST) -> bool:
        """Check if a node should be processed (is AstExtension)."""
        return isinstance(node, AstExtension)
    
    def should_visualize(self, node: ast.AST) -> bool:
        """Check if a node should be drawn in the graph."""
        return isinstance(node, AstExtension) and getattr(node, "_visualize", False)
    
    def format_io_list(self, io_list: list[ast.AST]) -> str:
        """Convert a list of AST nodes to a comma-separated string."""
        if not io_list:
            return ""
        
        strs = []
        for item in io_list:
            if isinstance(item, ast.AST):
                try:
                    strs.append(ast.unparse(item))
                except Exception:
                    strs.append(str(item))
            else:
                strs.append(str(item))
        return ", ".join(strs)
    
    def create_node_label(self, node: ast.AST) -> str:
        """
        Generate the display label for a node using HTML-like table format.
        Inputs are shown above, node name in the middle, outputs below.
        """
        # Get node name and tag
        node_name = type(node).__name__
        
        tag = getattr(node, "_tag", None)
        if tag is not None:
            if isinstance(tag, ast.AST):
                try:
                    tag_str = ast.unparse(tag)
                except Exception:
                    tag_str = ast.dump(tag)
            else:
                tag_str = str(tag)
            tag_str = tag_str.replace("\n", " ")
            node_name += f" [{tag_str}]"
        
        # Get inputs and outputs (always lists)
        inputs = getattr(node, "_inputs", [])
        outputs = getattr(node, "_outputs", [])
        
        inputs_str = self.format_io_list(inputs)
        outputs_str = self.format_io_list(outputs)
        
        # Choose color based on node type
        if isinstance(node, ast.For):
            main_bgcolor = "#ffd699"  # Orange for loops
        else:
            main_bgcolor = "#a5d8ff"  # Blue for other nodes
        
        # Build HTML-like table label
        # Structure: inputs (top row), node name (middle), outputs (bottom row)
        rows = []
        
        # Add inputs row if present
        if inputs_str:
            rows.append(f'<TR><TD BGCOLOR="#e3f2fd" BORDER="0"><FONT POINT-SIZE="9">in: {inputs_str}</FONT></TD></TR>')
        
        # Add main node name row
        rows.append(f'<TR><TD BGCOLOR="{main_bgcolor}" BORDER="0"><B>{node_name}</B></TD></TR>')
        
        # Add outputs row if present
        if outputs_str:
            rows.append(f'<TR><TD BGCOLOR="#e8f5e9" BORDER="0"><FONT POINT-SIZE="9">out: {outputs_str}</FONT></TD></TR>')
        
        # Combine into HTML table
        html_label = f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">{" ".join(rows)}</TABLE>>'
        
        return html_label
    
    def get_target_label(self, node: ast.AST) -> str | None:
        """Extract target variable names from assignment/loop nodes."""
        # Single target (For/AsyncFor)
        if hasattr(node, "target"):
            t = getattr(node, "target")
            if isinstance(t, ast.AST):
                try:
                    return ast.unparse(t).replace("\n", " ")
                except Exception:
                    return ast.dump(t)
        
        # Multiple targets (Assign)
        if hasattr(node, "targets"):
            ts = getattr(node, "targets", [])
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
    
    def visit_node_with_context(self, node: ast.AST, parent_id: str | None, edge_label: str | None) -> str | None:
        """
        Visit a node with specific parent and edge label context.
        Returns the ID of the last drawn node (or None).
        """
        # Skip nodes that are not AstExtension
        if not self.should_process(node):
            return None
        
        saved_parent = self.parent_id
        saved_edge_label = self.edge_label
        saved_last_id = self.last_id
        
        self.parent_id = parent_id
        self.edge_label = edge_label
        self.last_id = None
        
        self.visit(node)
        result_id = self.last_id
        
        self.parent_id = saved_parent
        self.edge_label = saved_edge_label
        self.last_id = saved_last_id
        
        return result_id
    
    def generic_visit(self, node: ast.AST) -> None:
        """
        Default handler - skip nodes that are not AstExtension.
        For AstExtension nodes that don't match Module/For, traverse children.
        """
        if not self.should_process(node):
            return
        
        # For AstExtension nodes without special handlers, just traverse children
        current_parent = self.parent_id
        
        for child in ast.iter_child_nodes(node):
            if self.should_process(child):
                self.visit_node_with_context(child, current_parent, None)
        
        self.last_id = self.parent_id
    
    def visit_Module(self, node: ast.Module) -> None:
        """Handle Module nodes - treat as a single visualized block."""
        if not self.should_process(node):
            return
        
        should_draw = self.should_visualize(node)
        my_id = None
        
        if should_draw:
            my_id = str(self.counter)
            self.counter += 1
            self.visible_ids.append(my_id)
            
            label = self.create_node_label(node)
            self.dot.node(my_id, label=label, shape="plaintext")
            
            if self.parent_id is not None:
                if self.edge_label is not None:
                    self.dot.edge(self.parent_id, my_id, label=self.edge_label)
                else:
                    self.dot.edge(self.parent_id, my_id)
        
        current_parent = my_id if my_id is not None else self.parent_id
        
        # Visit body statements (chain them sequentially)
        prev_id = current_parent
        last_child_id: str | None = None
        
        for stmt in node.body:
            if self.should_process(stmt):
                child_id = self.visit_node_with_context(stmt, prev_id, None)
                if child_id is not None:
                    prev_id = child_id
                    last_child_id = child_id
        
        # Set last_id for chaining
        self.last_id = last_child_id if last_child_id is not None else (my_id if should_draw else self.parent_id)
    
    def visit_For(self, node: ast.For) -> None:
        """Handle For-loops with dataflow-style chaining."""
        if not self.should_process(node):
            return
        
        should_draw = self.should_visualize(node)
        my_id = None
        
        if should_draw:
            my_id = str(self.counter)
            self.counter += 1
            self.visible_ids.append(my_id)
            
            label = self.create_node_label(node)
            # For For-loops, we keep using diamond shape but with HTML label
            self.dot.node(my_id, label=label, shape="plaintext")
            
            if self.parent_id is not None:
                if self.edge_label is not None:
                    self.dot.edge(self.parent_id, my_id, label=self.edge_label)
                else:
                    self.dot.edge(self.parent_id, my_id)
        
        current_parent = my_id if my_id is not None else self.parent_id
        
        # Chain body statements with edge labels
        prev_id = current_parent
        last_body_id: str | None = None
        label_owner: ast.AST | None = node
        
        for stmt in node.body:
            if not self.should_process(stmt):
                continue
            
            label_for_edge = self.get_target_label(label_owner) if label_owner is not None else None
            child_id = self.visit_node_with_context(stmt, prev_id, label_for_edge)
            
            if child_id is not None:
                prev_id = child_id
                last_body_id = child_id
                
                if not isinstance(stmt, ast.For):
                    label_owner = stmt
        
        # Add loop-back edge
        if should_draw and last_body_id is not None and my_id is not None and last_body_id != my_id:
            self.dot.edge(last_body_id, my_id, constraint="false")
        
        # Set effective last ID for parent-level chaining
        effective_last_id = last_body_id if last_body_id is not None else (my_id if should_draw else self.parent_id)
        self.last_id = effective_last_id


def _ast_to_graph(
    node: ast.AST,
    dot=None,
    parent_id: str | None = None,
    counter: list[int] | None = None,
    visible_ids: list[str] | None = None,
    edge_label: str | None = None,
):
    """
    Build a Graphviz graph for the AST using a visitor pattern.
    
    Only visualizes nodes that are:
      - instances of AstExtension, and
      - have _visualize set to True.
    """
    visitor = AstGraphVisitor(dot=dot)
    
    if counter is not None:
        visitor.counter = counter[0]
    if visible_ids is not None:
        visitor.visible_ids = visible_ids
    
    visitor.parent_id = parent_id
    visitor.edge_label = edge_label
    visitor.visit(node)
    
    if counter is not None:
        counter[0] = visitor.counter
    
    return visitor.dot, visitor.last_id, visitor.visible_ids


class AstExtension: 
    _fields: tuple[str, ...]
    

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

    class Wrapper(AstExtension, cls):
        pass

    Wrapper.__name__ = cls.__name__
    rv = typing.cast("type[ast.AST]", Wrapper)
    _to_extended[cls] = rv
    return rv

def create(cls, **fields: object):
    result = get_wrapper_cls(cls)(**fields)
    assert isinstance(result, AstExtension)
    return result
        