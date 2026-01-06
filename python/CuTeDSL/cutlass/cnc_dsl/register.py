import threading
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type
import importlib


class OverrideRegistry:
    """Global registry for tracking class extensions and function overrides."""

    def __init__(self) -> None:
        # List of extension info dicts:
        #   { "target_cls": type, "extension_cls": type, "methods": {name: attr} }
        self.extensions: List[Dict[str, Any]] = []

        # Function overrides map original callable -> override callable.
        # Wrapper-style dispatch consults this when a context is active.
        self.function_overrides: Dict[Callable[..., Any], Callable[..., Any]] = {}

    def register(
        self, target_cls: Type[Any], extension_cls: Type[Any], methods: Dict[str, Any]
    ) -> None:
        """Register an extension with its target class and methods."""
        self.extensions.append(
            {
                "target_cls": target_cls,
                "extension_cls": extension_cls,
                "methods": methods,  # Dict of {name: attr}
            }
        )

    def get_all_extensions(self) -> List[Dict[str, Any]]:
        """Get all registered extensions."""
        return self.extensions

    def register_func_override(
        self, original_func: Callable[..., Any], new_func: Callable[..., Any]
    ) -> None:
        """
        Register a function override for wrapper-style dispatch.

        Both functions MUST have the same signature.
        """
        self.function_overrides[original_func] = new_func

    def get_override_for(
        self, original_func: Callable[..., Any]
    ) -> Optional[Callable[..., Any]]:
        """Return the override registered for ``original_func``, if any."""
        return self.function_overrides.get(original_func)


# Global registry instance
_extension_registry = OverrideRegistry()

# Thread-local storage for active context
_active_context = threading.local()


class CnCOverrideContext:
    """Context manager to temporarily enable all registered extensions"""

    def __init__(self, registry=None):
        self.registry = registry or _extension_registry
        self.modifications = []  # Track class method modifications: (target_cls, name, original_attr)
        self.function_modifications = []  # Track function replacements: (target, attr_name, original_func)

    @classmethod
    def current(cls):
        """
        Get the currently active ExtensionContext.

        Returns None if no context is active.

        Usage:
            if ExtensionContext.current() is not None:
                # Context is active
                pass
        """
        return getattr(_active_context, "current", None)

    def __enter__(self):
        """Enable all extension methods when entering context."""
        # Set this as the active context
        _active_context.current = self

        # Apply class method extensions
        for ext_info in self.registry.get_all_extensions():
            target_cls = ext_info["target_cls"]
            extension_cls = ext_info["extension_cls"]
            methods = ext_info["methods"]

            for name, attr in methods.items():
                # Store original method if it exists
                if hasattr(target_cls, name):
                    original_attr = getattr(target_cls, name)
                    # Save original method with backup name so extension can access it
                    backup_name = f"_{target_cls.__name__}_{name}"
                    setattr(target_cls, backup_name, original_attr)
                    self.modifications.append((target_cls, name, original_attr))
                else:
                    # Mark as new method (None means it didn't exist)
                    self.modifications.append((target_cls, name, None))

                # Add the extension method
                setattr(target_cls, name, attr)

            # call extension hook, if defined
            hook = getattr(extension_cls, "_cnc__enter__", None)
            if callable(hook):
                hook(target_cls)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disable extension methods and function overrides when exiting context"""
        # Clear the active context
        _active_context.current = None

        # Restore replaced functions in reverse order
        for target, attr_name, original_func in reversed(self.function_modifications):
            if original_func is not None:
                # Restore original function
                setattr(target, attr_name, original_func)
            else:
                # Remove the function (it didn't exist before)
                if hasattr(target, attr_name):
                    delattr(target, attr_name)

        self.function_modifications.clear()

        # Restore class methods in reverse order
        for target_cls, name, original_attr in reversed(self.modifications):
            if original_attr is not None:
                # Restore original method
                setattr(target_cls, name, original_attr)
                # Remove the backup method
                backup_name = f"_{target_cls.__name__}_{name}"
                if hasattr(target_cls, backup_name):
                    delattr(target_cls, backup_name)
            else:
                # Remove the method entirely (it was added by extension)
                if hasattr(target_cls, name):
                    delattr(target_cls, name)

        self.modifications.clear()

        # Call extension cls exit hooks
        for ext_info in self.registry.get_all_extensions():
            target_cls = ext_info["target_cls"]
            extension_cls = ext_info["extension_cls"]
            hook = getattr(extension_cls, "_cnc__exit__", None)
            if callable(hook):
                hook(target_cls)
        return False  # Don't suppress exceptions


def register_cls_extension(target_cls):
    """
    Decorator to register methods from an extension class.

    Usage:
        @register_cls_extension(BaseCls)
        class ClsExtension:
            def new_method(self):
                pass

            def existing_method(self):
                # Can access original method via _BaseCls_existing_method
                original_result = self._BaseCls_existing_method()
                return f"Extended: {original_result}"

    The extension methods are NOT applied by default. They must be enabled using
    the ExtensionContext context manager:

        with ExtensionContext():
            # Extensions are enabled here
            BaseCls().new_method()  # Works!

    If an extension method overrides an existing method in the target class, the
    original method is saved as _<TargetClass>_<method_name> and can be accessed
    by the extension method.

    Multiple extensions can be registered and they will all be enabled/disabled together.
    """

    def decorator(extension_cls):
        methods: Dict[str, Any] = {}  # Track methods for this extension

        # Iterate through all attributes of the extension class
        for name, attr in extension_cls.__dict__.items():
            # Skip special/private methods
            if name.startswith("_cnc_"):
                continue

            # Include normal functions / callables
            if callable(attr):
                methods[name] = attr
                continue

            # Include descriptors like @staticmethod and @classmethod
            if isinstance(attr, (staticmethod, classmethod)):
                methods[name] = attr
                continue

        # Register this extension in the global registry
        # Extensions are NOT applied to target_cls yet
        _extension_registry.register(target_cls, extension_cls, methods)

        # Return the extension class (keeps it available if needed)
        return extension_cls

    return decorator


def register_func_override(original_func):
    """
    Decorator to register a function override.

    The new function MUST have the same signature as the original.

    Usage (wrapper-style):

        def original_func(x, y):
            return x + y

        @register_func_override(original_func)
        def wrapped(x, y):  # Same signature!
            return x * y

        # Outside of context: behaves like original_func
        wrapped(2, 3)  # -> 5

        with CnCOverrideContext():
            # Inside context: behaves like override
            wrapped(2, 3)  # -> 6

    Args:
        original_func: The function to override

    Returns:
        Decorator that registers the override and returns a wrapper function.
    """

    def decorator(new_func: Callable[..., Any]) -> Callable[..., Any]:
        # Record this override in the global registry
        _extension_registry.register_func_override(original_func, new_func)

        @wraps(original_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Wrapper that dispatches to either the original function or the override,
            depending on whether a CnCOverrideContext is active.
            """
            ctx = CnCOverrideContext.current()
            if ctx is not None:
                override = ctx.registry.get_override_for(original_func)
                if override is not None:
                    return override(*args, **kwargs)
            return original_func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["CnCOverrideContext"]


def _eager_import_extensions():
    from . import _CNC_OVERRIDE_MODULES

    for mod in _CNC_OVERRIDE_MODULES:
        importlib.import_module(mod, package=__package__)


_eager_import_extensions()
