from .graph import KernelLoopState, CnCContext
from ..base_dsl.ast_helpers import cf_symbol_check, get_locals_or_none


def loop_selector(
    start,
    stop,
    step,
    *,
    write_args=[],
    full_write_args_count=0,
    write_args_names=[],
    unroll=-1,
    unroll_full=False,
    prefetch_stages=None,
):
    def decorator(func):
        # pass function pointer into KernelLoopState
        loop_state = KernelLoopState.create_from_loop_selector(
            func,
            start,
            stop,
            step,
            write_args,
            full_write_args_count,
            write_args_names,
            unroll,
            unroll_full,
            prefetch_stages,
        )
        CnCContext.current().register_loop(loop_state)

        from ..base_dsl.ast_helpers import loop_selector as base_loop_selector

        # call the original base decorator on func
        base_dec = base_loop_selector(
            start,
            stop,
            step,
            write_args=write_args,
            full_write_args_count=full_write_args_count,
            write_args_names=write_args_names,
            unroll=unroll,
            unroll_full=unroll_full,
            prefetch_stages=prefetch_stages,
        )
        ret = base_dec(func)
        CnCContext.current().current_kernel().exit_loop()
        return ret

    return decorator


__all__ = ["loop_selector", "cf_symbol_check", "get_locals_or_none"]
