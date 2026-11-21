from __future__ import annotations

from contextlib import contextmanager


class FrozenModel:
    """Legacy compatibility helper; the common end-to-end path never constructs it."""
    def __init__(self, path, config, actor, updater=None):
        import copy
        import torch
        from models.hf import HFPolicy
        self.torch, self.actor, self.updater = torch, actor, updater
        cfg = copy.deepcopy(config)
        cfg['model'].update(device='cpu', cpu_dtype=cfg['model']['dtype'])
        self.policy = HFPolicy(path, cfg)
        self.policy.model.eval().requires_grad_(False)
        self.device = config['model'].get('answer_device', 'cuda:0')
        self.depth = 0

    @contextmanager
    def phase(self):
        if self.depth:
            yield
            return
        self.depth += 1
        original = getattr(self.actor, 'device', None)
        try:
            if hasattr(self.actor, 'offload_all'):
                self.actor.offload_all()
            if hasattr(self.actor, 'model'):
                self.actor.model.cpu()
                if self.updater:
                    for values in self.updater.optimizer.state.values():
                        for key, value in values.items():
                            if self.torch.is_tensor(value) and key != 'step':
                                values[key] = value.cpu()
            # The distributed actor is configured with parameter/gradient/optimizer offload;
            # its vLLM engine sleeps on leaving generation (inherited sharding manager).
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            self.policy.model.to(self.device)
            self.policy.device = self.torch.device(self.device)
            yield
        finally:
            self.policy.model.cpu()
            self.policy.device = self.torch.device('cpu')
            # Ray actors have separate allocators; they cannot reclaim this driver's HF cache.
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            if hasattr(self.actor, 'model'):
                self.actor.model.to(original)
                if self.updater:
                    for values in self.updater.optimizer.state.values():
                        for key, value in values.items():
                            if self.torch.is_tensor(value) and key != 'step':
                                values[key] = value.to(original)
            self.depth -= 1

    def set_seed(self, seed):
        self.policy.set_seed(seed)

    def complete(self, *args, **kwargs):
        with self.phase():
            return self.policy.complete(*args, **kwargs)
