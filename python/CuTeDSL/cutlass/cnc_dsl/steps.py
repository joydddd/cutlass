from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
from types import FrameType
import inspect


import cutlass.cute as cute
from ..base_dsl.dsl import BaseDSL
from ..base_dsl.utils.logger import log

from .tag import Tag
from .trace import Context


@dataclass
class Step:
    _name: str
    _func: Callable
    _decorator_frame: FrameType
    _tag_name: str | None
    _static_tag: Optional[Tag] = None
    _id: Optional[int] = None

    def __init__(
        self,
        name: str,
        func: Callable,
        tag: Tag | str,
        decorator_frame: FrameType,
        step_id: Optional[int] = None,
    ):
        self._name = name
        self._func = func
        self._decorator_frame = decorator_frame
        if isinstance(tag, str):
            self._tag_name = tag
        else:
            self._tag_name = None
            self._static_tag = tag
        self._step_id = step_id

    def __repr__(self):
        return f"<Step> {self.name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def func(self) -> Callable:
        return self._func

    @property
    def id(self) -> int:
        if self._id is None:
            raise ValueError("Step id is not set")
        return self._id

    def set_id(self, id: int) -> None:
        if self._id is not None:
            raise ValueError("Step id is already set")
        self._id = id

    def _retrive_tag_from_args(self, *args, **kwargs):
        if self._static_tag:  # Use static tag.
            return self._static_tag
        if self._tag_name in kwargs:
            return kwargs[self._tag_name]

        sig = inspect.signature(self._func)
        param_list = list(sig.parameters.keys())
        if self._tag_name in param_list:
            return args[param_list.index(self._tag_name)]
        else:
            raise ValueError(
                f"Tag {self._tag_name} is not a parameter of {self._func.__name__}"
            )

    def __call__(self, *args, **kwargs):
        lifter_ctx = Context.current()
        if lifter_ctx is not None:
            if self._static_tag is not None:
                lifter_ctx.register_static_tag(self._static_tag)
            tag = self._retrive_tag_from_args(*args, **kwargs)
            lifter_ctx.register_step(self, tag)

            # Trace into the step.
            return self.func(*args, **kwargs)
        else:
            # CnCLifter not enabled. Continue the normal jit compilation.
            return self.func._jit_wrapper(*args, **kwargs)  # type: ignore[attr-defined]


def step(tag: str | Tag, jit: Callable = cute.jit, *jit_args, **jit_kwargs) -> Callable:
    """
    Decorator to register a function as a CnC step and JIT compile it.

    Usage:
        @cnc.step(tag="x")
        def my_step(x, y):
            return x + y

        @cnc.step(tag="y", preprocessor=False)
        def my_step(x, y):
            return x + y

    Args:
        tag: The tag to associate with this step
        *jit_args: Positional arguments to pass through to @cute.jit
    """

    if tag is None:
        raise ValueError("tag must be provided for cnc.step. ")

    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("Cannot determine step decorator frame")
    frame = frame.f_back

    if not hasattr(jit, "__self__"):
        raise ValueError(f"jit must be a method of a DSL class, but got {type(jit)}")
    dsl_cls = jit.__self__
    if not issubclass(dsl_cls, BaseDSL):
        raise ValueError(
            f"jit must be a method of a DSL class inherited from BaseDSL, but got {type(dsl_cls)}"
        )

    jit_decorator = dsl_cls.jit_runner(dsl_cls, "_func", frame, *jit_args, **jit_kwargs)

    log().info("cnc node")

    def step_decorator(func: Callable) -> Callable:
        print(f"Processing step: {func.__name__}")

        # Apply the jit decorator with pass-through arguments
        # Handle both @jit and @jit(...) forms
        jit_wrapper = jit_decorator(func)
        func._jit_wrapper = jit_wrapper  # type: ignore[attr-defined]

        return Step(
            name=func.__name__,
            func=func,
            tag=tag,
            decorator_frame=frame,
        )

    return step_decorator
