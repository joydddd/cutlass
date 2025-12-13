from __future__ import annotations

import unittest

from cutlass.cnc_dsl.register import (
    CnCOverrideContext,
    register_cls_extension,
    register_func_override,
)


# Helper state for function override test
CALLS: list[str] = []


class TestCnCRegister(unittest.TestCase):
    def test_cls_extension_adds_and_restores_methods(self) -> None:
        class Base:
            def __init__(self, x: int) -> None:
                self.x = x

            def inc(self) -> int:
                return self.x + 1

        @register_cls_extension(Base)
        class BaseExtension:
            def __init__(self, x: int) -> None:
                # Call original Base.__init__ saved under _Base___init__
                self._Base___init__(x)  # type: ignore[attr-defined]
                self.extra = 10

            def inc(self) -> int:
                # Access original Base.inc saved under _Base_inc
                return self._Base_inc() + self.extra  # type: ignore[attr-defined]

        # Outside of context: original behavior only
        b = Base(5)
        self.assertEqual(b.inc(), 6)
        self.assertFalse(hasattr(b, "extra"))

        with CnCOverrideContext():
            # Inside context: extension behavior is active
            b_ctx = Base(5)
            self.assertEqual(b_ctx.extra, 10)
            self.assertEqual(b_ctx.inc(), 16)

        # After context: original behavior restored
        b_after = Base(5)
        self.assertFalse(hasattr(b_after, "extra"))
        self.assertEqual(b_after.inc(), 6)

    def test_function_override_applies_only_in_context(self) -> None:
        def original(x: int, y: int) -> int:
            CALLS.append("orig")
            return x + y

        @register_func_override(original)
        def wrapped(x: int, y: int) -> int:
            CALLS.append("new")
            return x * y

        # Outside context: behaves like original
        CALLS.clear()
        self.assertEqual(wrapped(2, 3), 5)
        self.assertEqual(CALLS, ["orig"])

        # Inside context: behaves like override
        CALLS.clear()
        with CnCOverrideContext():
            self.assertEqual(wrapped(2, 3), 6)
            self.assertEqual(CALLS, ["new"])

        # After context: original behavior again
        CALLS.clear()
        self.assertEqual(wrapped(2, 3), 5)
        self.assertEqual(CALLS, ["orig"])

    def test_current_tracks_active_context_and_reset(self) -> None:
        # No active context initially
        self.assertIsNone(CnCOverrideContext.current())

        with CnCOverrideContext() as ctx:
            self.assertIs(CnCOverrideContext.current(), ctx)

        # After all contexts exit, current should be None
        self.assertIsNone(CnCOverrideContext.current())


if __name__ == "__main__":
    unittest.main()


