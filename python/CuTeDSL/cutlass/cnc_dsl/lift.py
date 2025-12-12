from ..base_dsl.dsl import DSLRuntimeError
from ..base_dsl.compiler import EmptyCompileOption, BooleanBasedFileDumpOption
import inspect

from typing import Callable

class DumpDir(EmptyCompileOption):
    option_name = "dump-dir"


class GenGraph(BooleanBasedFileDumpOption):
    option_name = "dump-graph-path"


class LiftOptions:
    def __init__(self, options=None):
        self.options = {
            DumpDir: DumpDir(""),
            GenGraph: GenGraph(False),
        }
        
        if options is not None: 
            self._update(options)
    
    def apply_settings(self, function_name: str):
        if self.options[GenGraph].value:
            self.options[GenGraph].dump_path = os.path.join(
                self.options[DumpDir].value, function_name
            )
    

    def _update(self, options):
        def _validate_and_update_option(option):
            if type(option) not in self.options:
                raise DSLRuntimeError(f"Invalid compile option: {option}")
            self.options[type(option)] = option

        if isinstance(options, tuple):
            for option in options:
                _validate_and_update_option(option)
        else:
            _validate_and_update_option(options)
    
    @property
    def dump_graph_path(self) -> str | None:
        return (
            self.options[GenGraph].full_graph_path if self.options[GenGraph].value else None
        )


class LiftCallable:
    def __init__(self, options=None) -> None:
        def preprocess_options(option):
            if type(option) is type and issubclass(
                option, (BooleanBasedFileDumpOption)
            ):
                # Automatically creates a True instance of the option
                return option(True)
            elif isinstance(option, tuple):
                return tuple(preprocess_options(opt) for opt in option)
            return option
        
        self._lift_options = LiftOptions(preprocess_options(options))


    def __call__(self, *args, **kwargs):
        return self._lift(*args, **kwargs)

    def _lift(self, target: Callable | classmethod | type, *args, **kwargs):
        if inspect.isfunction(target):
            func = target
        elif inspect.ismethod(target):
            args = [target.__self__] + list(args)
            func = target.__func__
        elif (
            inspect.isclass(type(target))
            and hasattr(target, "__call__")
            and hasattr(target.__call__, "__func__")
        ): 
            # if target is a class instance, compile class's __call__ method
            args = [target] + list(args)
            func = target.__call__.__func__

        # If it's a wrapped function created by jit decorator, get the original function
        if hasattr(func, "__wrapped__"):
            func = func.__wrapped__ 

        from ..base_dsl.dsl import BaseDSL

        BaseDSL._lazy_initialize_dsl(func)

        if not hasattr(func, "_dsl_object"):
            raise DSLRuntimeError("Function is not decorated with jit decorator.")


        # What to preprocess here? 

        # Remove all nodes in ast graph, except for the For & If nodes. define consume and produce node for each step. 
        


        func_ptr = func._dsl_object._preprocess_and_execute(func)


        if hasattr(func, "_decorator_frame"):
            kwargs["_decorator_frame"] = func._decorator_frame
        return func._dsl_object._func(func_ptr, *args, **kwargs)