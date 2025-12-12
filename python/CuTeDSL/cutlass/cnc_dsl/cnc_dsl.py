from .register import register_cls_extension, ExtensionContext
from ..base_dsl.dsl import BaseDSL
from typing import Callable


@register_cls_extension(BaseDSL)
class cncDSLExtension:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.envar.dryrun = True

    @staticmethod
    def run_cnc_analysis(func: Callable): 
        print("Run ast analysis for function: ", func.__name__)
    

    @staticmethod
    def _preprocess_and_execute(cls, func: Callable) -> Callable:
        cls.run_cnc_analysis(func)
        return func
    

    @classmethod
    def _get_original_function(cls, fcn_ptr, name):
        print(f"In cncDSLExtension._get_original_function for function: {name}")
        breakpoint()
        while hasattr(fcn_ptr, "__wrapped__"):
            fcn_ptr = fcn_ptr.__wrapped__
            fcn_ptr = cls._BaseDSL__get_original_function(fcn_ptr, name)
        return fcn_ptr
