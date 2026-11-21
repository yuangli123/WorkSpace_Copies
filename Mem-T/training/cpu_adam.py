"""Make CPUAdam's unsaved scalar bias cache independent of process history."""
from functools import wraps
from types import MethodType


def normalize_bias_cache(optimizer):
    if optimizer is None or not all(hasattr(optimizer, key) for key in ('ds_opt_adam', 'opt_id', 'adam_w_mode')):
        return False
    if getattr(optimizer, '_memt_bias_cache_normalized', False):
        return True
    original_step = optimizer.step

    @wraps(original_step)
    def step(self, *args, **kwargs):
        group = next((g for g in self.param_groups if g['params']), self.param_groups[0])
        beta1, beta2 = group['betas']
        # CPUAdam IncrementStep otherwise multiplies cached beta powers during
        # uninterrupted training but uses pow(beta, step) after loading. That C++
        # cache is absent from state_dict; a BF16 rounding boundary exposed the
        # mismatch in the four-rank restore test. Recreate only the scalar kernel
        # object so both paths use the same power calculation. Python moments,
        # FP32 master parameters, counters, groups, and scheduler hooks are kept.
        self.ds_opt_adam.destroy_adam(self.opt_id)
        self.ds_opt_adam.create_adam(self.opt_id, group['lr'], beta1, beta2, group['eps'],
                                    group['weight_decay'], self.adam_w_mode, False)
        return original_step(*args, **kwargs)

    optimizer.step = MethodType(step, optimizer)
    optimizer._memt_bias_cache_normalized = True
    return True
