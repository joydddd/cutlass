from ..base_dsl.ast_preprocessor import DSLPreprocessor
import inspect
import ast
from ..base_dsl.utils.logger import log
from ..base_dsl.common import DSLRuntimeError
import textwrap
from typing import List


class CnCProcessor(DSLPreprocessor):
    def import_modules(self):
        """
        Import cnc_dsl as _dsl_ module. Then the _dsl_.loop_selector etc. will be avilable from cnc_dsl.ast_helper. 
        """
        top_module_name = ".".join(self.client_module_name) # 'import cutlass'
        import_stmts = []
        if self.import_top_module:
            import_stmts.append(ast.Import(names=[ast.alias(name=top_module_name)]))
        import_stmts.append(
            ast.Import(
                names=[ast.alias(name=f"{top_module_name}.cnc_dsl", asname="_dsl_")] 
            )
        )
        return import_stmts
        
    def transform_function(self, func_name, function_pointer) -> List[ast.Module]:
        tree = super().transform_function(func_name, function_pointer)
        return tree
    
    def exec(self, function_name, original_function, code_object, exec_globals):
        return super().exec(function_name, original_function, code_object, exec_globals)