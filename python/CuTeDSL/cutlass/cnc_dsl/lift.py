from ..base_dsl.dsl import DSLRuntimeError
import inspect

class LiftCallable:
    def __init__(self, options=None):
        pass 


    def __call__(self, *args, **kwargs):
        return self._lift(*args, **kwargs)

    def _lift(self, func, *args, **kwargs):
        if func is None:
            raise DSLRuntimeError("Function is not set or invalid.")

        if not callable(func):
            raise DSLRuntimeError("Object is not callable.")
        
        if not inspect.isfunction(func):
            raise DSLRuntimeError("Function is not a regular function.")
        


        # If it's a wrapped function created by jit decorator, get the original function
        if hasattr(func, "__wrapped__"):
            func = func.__wrapped__ 
        

        from ..base_dsl.dsl import BaseDSL

        BaseDSL._lazy_initialize_dsl(func)

        if not hasattr(func, "_dsl_object"):
            raise DSLRuntimeError("Function is not decorated with jit decorator.")


        func_ptr = func._dsl_object._preprocess_and_execute(func)


        if hasattr(func, "_decorator_frame"):
            kwargs["_decorator_frame"] = func._decorator_frame
        return func._dsl_object._func(func_ptr, *args, **kwargs)