from __future__ import annotations

import ast
from typing import List

from ..base_dsl.ast_preprocessor import DSLPreprocessor


class CnCProcessor(DSLPreprocessor):
    def _import_modules(self):
        """
        Import cnc_dsl as _dsl_ module. Then the _dsl_.loop_selector etc. will be avilable from cnc_dsl.ast_helper.
        """
        top_module_name = ".".join(self.client_module_name)
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
