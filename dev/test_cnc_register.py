from cutlass.cnc_dsl.register import * 
"""
Test function overrides with CnCOverrideContext.
"""

# ---- Class extension tests ----

class Base:
    def foo(self):
        return "base-foo"


@register_cls_extension(Base)
class BaseExtension:
    def bar(self):
        return "bar-from-extension"

    def foo(self):
        # Access original via backup name installed by the context
        return f"ext-{self._Base_foo()}"


def test_class_extension_applied_and_restored():
    b = Base()

    # Before context: no extension behavior
    assert CnCOverrideContext.current() is None
    assert b.foo() == "base-foo"
    print(f"b.foo(): {b.foo()} before context")
    assert not hasattr(b, "bar")

    with CnCOverrideContext() as ctx:
        # current() should see this context
        assert CnCOverrideContext.current() is ctx

        # Extension methods are now visible
        assert b.foo() == "ext-base-foo"
        assert b.bar() == "bar-from-extension"

        print(f"b.foo(): {b.foo()} in context")
        print(f"b.bar(): {b.bar()} in context")
    
    print(f"b.foo(): {b.foo()} outside of context")

    # After context: methods are restored
    assert CnCOverrideContext.current() is None
    assert b.foo() == "base-foo"
    assert not hasattr(b, "bar")


# ---- Function override tests ----

def target_func(x, y):
    return x + y


@register_function_override(target_func)
def target_func_override(x, y):
    # Same signature, different behavior
    return x * y


def test_function_override_applied_and_restored():
    # Before context: original behavior
    assert CnCOverrideContext.current() is None
    assert target_func(2, 3) == 5
    print(f"target_func(2, 3): {target_func(2, 3)} before context")


    with CnCOverrideContext():
        # Inside context: override is active
        assert target_func(2, 3) == 6
        print(f"target_func(2, 3): {target_func(2, 3)} in context")


    # After context: original behavior restored
    assert target_func(2, 3) == 5
    print(f"target_func(2, 3): {target_func(2, 3)} outside of context")



test_class_extension_applied_and_restored()
test_function_override_applied_and_restored()