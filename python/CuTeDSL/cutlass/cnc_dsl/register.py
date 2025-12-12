import threading

class OverrideRegistry:
    """Global registry for tracking class extensions and function overrides"""
    
    def __init__(self):
        self.extensions = []  # List of extension info dicts
        self.function_overrides = []  # List of function override info dicts
    
    def register(self, target_cls, extension_cls, methods):
        """Register an extension with its target class and methods"""
        self.extensions.append({
            'target_cls': target_cls,
            'extension_cls': extension_cls,
            'methods': methods  # Dict of {name: attr}
        })
    
    def get_all_extensions(self):
        """Get all registered extensions"""
        return self.extensions
    
    def register_function_override(self, new_func, target, attr_name):
        """
        Register a function override.
        
        The new function MUST have the same signature as the original.
        
        Args:
            new_func: The new function (same signature as original)
            target: The module/class where the original function lives
            attr_name: The attribute name on target
        """
        self.function_overrides.append({
            'new_func': new_func,
            'target': target,
            'attr_name': attr_name
        })
    
    def get_all_function_overrides(self):
        """Get all registered function overrides"""
        return self.function_overrides


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
        return getattr(_active_context, 'current', None)
    
    def __enter__(self):
        """Enable all extension methods and function overrides when entering context"""
        # Set this as the active context
        _active_context.current = self
        
        # Apply class method extensions
        for ext_info in self.registry.get_all_extensions():
            target_cls = ext_info['target_cls']
            methods = ext_info['methods']
            
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
        
        # Apply function overrides from registry
        for override_info in self.registry.get_all_function_overrides():
            new_func = override_info['new_func']
            target = override_info['target']
            attr_name = override_info['attr_name']
            
            # Store the original function from target
            if hasattr(target, attr_name):
                original_func = getattr(target, attr_name)
            else:
                original_func = None
            
            self.function_modifications.append((target, attr_name, original_func))
            
            # Replace with new function (same signature as original)
            setattr(target, attr_name, new_func)
        
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
        methods = {}  # Track methods for this extension
        
        # Iterate through all attributes of the extension class
        for name, attr in extension_cls.__dict__.items():
            # Skip special/private methods and non-callables
            if not name.startswith('_cnc_') and callable(attr):
                # Store the method (but don't apply it yet)
                methods[name] = attr
        
        # Register this extension in the global registry
        # Extensions are NOT applied to target_cls yet
        _extension_registry.register(target_cls, extension_cls, methods)
        
        # Return the extension class (keeps it available if needed)
        return extension_cls
    
    return decorator


def register_function_override(original_func):
    """
    Decorator to register a function override.
    
    The new function MUST have the same signature as the original.
    
    Usage:
        def original_func(x, y):
            return x + y
        
        @register_function_override(original_func)
        def new_func(x, y):  # Same signature!
            return x * y
        
        with ExtensionContext():
            original_func(2, 3)  # Calls new_func, returns 6
    
    Args:
        original_func: The function to override
    
    Returns:
        Decorator that registers the override
    """
    def decorator(new_func):
        # Extract where the function lives
        target, attr_name = _extract_function_location(original_func)
        
        # Register: just store new_func, target, and attr_name
        # Original will be captured from target when context activates
        _extension_registry.register_function_override(
            new_func=new_func,
            target=target,
            attr_name=attr_name
        )
        
        return new_func
    
    return decorator


def _extract_function_location(func):
    """
    Extract the target and attribute name for a function.
    
    Returns (target, attr_name) tuple where:
    - target is the module or class where the function is defined
    - attr_name is the name of the attribute
    
    For module-level functions: (module, 'func_name')
    For methods: (class, 'method_name')
    """
    import sys
    
    # Get the module where the function is defined
    module_name = func.__module__
    if module_name in sys.modules:
        module = sys.modules[module_name]
        
        # Check if it's a simple module-level function
        qualname = func.__qualname__
        if '.' not in qualname:
            # Module-level function
            return (module, func.__name__)
        else:
            # It's a method or nested function
            # Try to resolve the class
            parts = qualname.split('.')
            attr_name = parts[-1]
            class_path = parts[:-1]
            
            # Navigate to the class
            target = module
            for part in class_path:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    # Can't resolve, fall back to module
                    return (module, func.__name__)
            
            return (target, attr_name)
    
    # Fallback: can't determine location
    return (None, func.__name__)


__all__ = ["register_cls_extension", "register_function_override", "CnCOverrideContext"]