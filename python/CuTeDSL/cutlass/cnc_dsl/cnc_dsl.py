from __future__ import annotations
from typing import Callable
import inspect
from functools import wraps

from .register import register_cls_extension
from .ast_processor import CnCProcessor
from .graph import KernelBuilder, CnCContext

from ..base_dsl.dsl import DSLRuntimeError
from ..cutlass_dsl.cutlass import CuTeDSL


@register_cls_extension(CuTeDSL)
class cncDSL(CuTeDSL):
    @staticmethod
    def _cnc__enter__(target_cls):
        """
        CnC hook invoked by `CnCOverrideContext` when entering the override scope.
        """
        # Clear the cache of _get_dsl method for CuTeDSL to force re-initialization.
        target_cls._get_dsl.__func__.cache_clear()

    @staticmethod
    def _cnc__exit__(target_cls):
        """
        CnC hook invoked by `CnCOverrideContext` when exiting the override scope.
        """
        target_cls._get_dsl()  # reinstall _get_dsl method for CuTeDSL.

    def __init__(self, *args, **kwargs):
        self._CuTeDSL___init__(*args, **kwargs)
        self.enable_preprocessor = True
        self.preprocessor = CnCProcessor(["cutlass"])
        self.envar.dryrun = True

    def kernel_launcher(self, *dargs, **dkwargs):
        """
        Overwrite kernel_launcher to create a register a new kernel builder everytime a kernel is launched.
        """
        original_decorator = self._CuTeDSL_kernel_launcher(*dargs, **dkwargs)

        def decorator(funcBody):
            @wraps(funcBody)
            def kernel_wrapper(*args, **kwargs):
                kernel_name = funcBody.__name__
                args_spec = inspect.getfullargspec(funcBody)
                # Give the kernel a unique name.
                kernel_name = f"kernel_{self.mangle_name(kernel_name, args, args_spec)}_{self.num_kernels}"
                if "config" in kwargs:
                    config = kwargs["config"]
                else:
                    raise DSLRuntimeError(
                        f"config is required for launching {kernel_name}."
                    )

                # remove kernel launcher args from kwargs similar to BaseDSL.kernel_launcher.
                requiredArgs = dkwargs.get("requiredArgs", [])
                optionalArgs = dkwargs.get("optionalArgs", [])
                processed_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in requiredArgs and k not in optionalArgs
                }

                kernel_builder = KernelBuilder.create_from_kernel_launch(
                    kernel_name, config, funcBody, *args, **processed_kwargs
                )
                CnCContext.current().register_kernel(kernel_builder)

                ret = original_decorator(funcBody)(*args, **kwargs)
                CnCContext.current().exit_kernel()
                return ret

            return kernel_wrapper

        return decorator

    def run_preprocessor(self, func: Callable):
        transformed_ast = self._CuTeDSL_run_preprocessor(func)
        # self.preprocessor.print_ast(transformed_ast)
        return transformed_ast

    @staticmethod
    def _preprocess_and_execute(func: Callable) -> Callable:
        ## TODO: clear all _dsl_object and create new cncDSL object instead.

        # Remove tranformed ast and Force re-preprocess.
        if hasattr(func, "_transformed_ast"):
            func._transformed_ast = None
            if hasattr(func, "_preprocessed"):
                delattr(func, "_preprocessed")
        return cncDSL._CuTeDSL__preprocess_and_execute(func)
