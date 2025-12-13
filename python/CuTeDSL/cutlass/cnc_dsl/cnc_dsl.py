from ..base_dsl.dsl import BaseDSL
from .register import register_cls_extension
from ..cutlass_dsl.cutlass import CuTeDSL
from typing import Callable
from .ast_processor import CnCProcessor


@register_cls_extension(CuTeDSL)
class cncDSL(CuTeDSL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_preprocessor = True
        self.preprocessor = CnCProcessor(["cutlass"])


    def run_preprocessor(self, func: Callable):
        print("Run ast analysis for function: ", func.__name__)
        ast = super().run_preprocessor(func)
        self.preprocessor.print_ast(ast)
        return ast

    # @staticmethod
    # def _preprocess_and_execute(func: Callable) -> Callable:
    #     breakpoint()
    #     cncDSL.run_cnc_analysis(func)
    #     return func

    @classmethod
    def _get_original_function(cls, fcn_ptr, name):
        print(f"In cncDSLExtension._get_original_function for function: {name}")
        breakpoint()
        while hasattr(fcn_ptr, "__wrapped__"):
            fcn_ptr = fcn_ptr.__wrapped__
            # call original function defined in BaseDSL.
            fcn_ptr = super()._BaseDSL__get_original_function(fcn_ptr, name)
        return fcn_ptr
