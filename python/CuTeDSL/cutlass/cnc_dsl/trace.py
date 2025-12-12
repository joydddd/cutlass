from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .steps import Step
    from .tag import Tag


class Origin:
    pass


class TagOrigin(Origin):
    tag_id: int

    def __init__(self, tag_id: int):
        self.tag_id = tag_id

    pass


class StepOrigin(Origin):
    step_id: int
    tag: Tag

    def __init__(self, step: int, tag: Tag):
        self.step = step
        self.tag = tag


class Context:
    _current: ClassVar["Context | None"] = None

    def __init__(self):
        super().__init__()
        self.step_registry: dict[int, Step] = {}  # step_id -> Step
        self.step_origin_registry: dict[
            int, Tuple[StepOrigin, ...]
        ] = {}  # step_id -> (StepOrigin, ...)
        self.tag_registry: dict[int, TagOrigin] = {}  # tag_id -> TagOrigin
        self._next_step_id: int = 0
        self._next_tag_id: int = 0

    def __enter__(self) -> "Context":
        # Save previous context to support nesting
        self._prev_context = Context._current
        Context._current = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Restore previous context when leaving the with-block
        Context._current = getattr(self, "_prev_context", None)

    @staticmethod
    def current() -> "Context | None":
        """
        Return the current active Context, or None if no Context is active.

        Intended usage:

            with Context():
                ctx = Context.current()
                assert ctx is not None
        """
        return Context._current

    def register_static_tag(self, tag: Tag) -> TagOrigin:
        tag_id = self._next_tag_id
        self._next_tag_id += 1
        self.tag_registry[tag_id] = TagOrigin(tag_id)
        return self.tag_registry[tag_id]

    def register_step(
        self, step: Step, tag: Tag, step_id: int | None = None
    ) -> StepOrigin:
        try:
            sid = step.id
        except ValueError:
            if step_id is None:
                step_id = self._next_step_id
                self._next_step_id += 1
            step.set_id(step_id)
            sid = step_id

        # Check duplicate step_id
        if sid in self.step_registry:
            raise ValueError(
                f"Duplicated Step id: Step id={sid} is already registered for {self.step_registry[sid]}"
            )

        # Check duplicate function pointer
        for existing in self.step_registry.values():
            if existing.func is step.func:
                raise ValueError(
                    f"Function {step.func} already registered for {existing}"
                )
        self.step_registry[sid] = step
        origin = StepOrigin(sid, tag)
        if sid not in self.step_origin_registry:
            self.step_origin_registry[sid] = (origin,)
        else:
            self.step_origin_registry[sid] = self.step_origin_registry[sid] + (origin,)
        return origin
