from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from types import FrameType
from typing import Optional


@dataclass(frozen=True)
class SourceLoc:
    """
    Lightweight description of a Python source location.

    This plays a similar role to Helion's ``SourceLocation.from_ast`` helper
    (see `helion/_compiler/source_location.py`), but instead of reading
    locations from an ``ast.AST`` node and Torch FX metadata, it derives them
    directly from Python stack frames via ``inspect``.
    """

    filename: str
    lineno: int
    colno: int = 0
    end_lineno: Optional[int] = None
    end_colno: Optional[int] = None
    name: Optional[str] = None

    # --- Constructors -----------------------------------------------------

    @staticmethod
    def from_frame(frame: FrameType) -> "SourceLoc":
        """
        Build a ``SourceLoc`` from an ``inspect`` frame.

        Mirrors the Python-version-sensitive logic in
        ``cutlass.base_dsl._mlir_helpers.op.dsl_user_op``:
        - On Python < 3.11, column/ending information is not available and
          we only record the starting line.
        - On Python >= 3.11, we use the richer ``positions`` API.
        """
        frame_info = inspect.getframeinfo(frame)

        filename = frame_info.filename
        func_name = frame_info.function

        positions = getattr(frame_info, "positions", None)
        if positions is not None:
            # Python >= 3.11: full positional information is available
            lineno = positions.lineno
            colno = positions.col_offset or 0
            end_lineno = positions.end_lineno
            end_colno = positions.end_col_offset
        else:
            # Python < 3.11: only the starting line number is known
            lineno = frame_info.lineno
            colno = 0
            end_lineno = None
            end_colno = None

        return SourceLoc(
            filename=filename,
            lineno=lineno,
            colno=colno,
            end_lineno=end_lineno,
            end_colno=end_colno,
            name=func_name,
        )

    @staticmethod
    def current(stacklevel: int = 1) -> "SourceLoc":
        """
        Convenience helper that records the caller's location.

        ``stacklevel=1`` gives the direct caller, ``stacklevel=2`` its caller,
        and so on.
        """
        frame = inspect.currentframe()
        # Skip this helper's own frame
        if frame is not None:
            frame = frame.f_back

        # Walk back ``stacklevel-1`` additional frames.
        for _ in range(stacklevel - 1):
            if frame is None:
                break
            frame = frame.f_back

        if frame is None:
            return SourceLoc("<unknown>", 0, 0, None, None, None)

        return SourceLoc.from_frame(frame)

    # --- Presentation helpers --------------------------------------------

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}"

    def __repr__(self) -> str:
        return f"<SourceLoc {os.path.basename(self.filename)}:{self.lineno}>"

    def short(self) -> str:
        """Return a short ``file:line`` representation."""
        return f"{os.path.basename(self.filename)}:{self.lineno}"
