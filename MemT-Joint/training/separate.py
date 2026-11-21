"""M5: same sampled lifecycle and objective, genuinely separate role parameters."""
from __future__ import annotations

import copy
from pathlib import Path

from experiments.io import read_json, write_json


class RoleBundle:
    def __init__(self, policies, device, updaters=None):
        self.policies, self.updaters, self.device = policies, updaters or {}, device
        self.tokenizer = next(iter(policies.values())).tokenizer

    def offload_all(self):
        import torch
        for policy in {id(p): p for p in self.policies.values()}.values():
            policy.model.cpu()
            policy.device = torch.device('cpu')
        for updater in self.updaters.values():
            for values in updater.optimizer.state.values():
                for key, value in values.items():
                    if torch.is_tensor(value) and key != 'step':
                        values[key] = value.cpu()

    def activate(self, role):
        import torch
        self.offload_all()
        policy = self.policies[role]
        policy.model.to(self.device)
        policy.device = torch.device(self.device)
        if role in self.updaters and not hasattr(self, 'common_config'):
            for values in self.updaters[role].optimizer.state.values():
                for key, value in values.items():
                    if torch.is_tensor(value) and key != 'step':
                        values[key] = value.to(self.device)
        return policy

    def generation_for(self, role):
        from models.vllm import VLLMPolicy
        policy = self.policies[role]
        if not hasattr(self, 'generation'):
            self.generation = VLLMPolicy(policy.model_path, self.common_config, policy, self.offload_all)
        if self.generation.owner is not policy:
            self.generation.owner = policy
            self.generation.invalidate()
        policy.generation = self.generation
        return self.generation


class SeparateUpdater:
    def __init__(self, bundle):
        self.bundle = bundle

    def update(self, rows):
        result = {'separate_parameters': True, 'optimizer_updates': 0}
        for role, updater in self.bundle.updaters.items():
            selected = [row for row in rows if row['role'] == role]
            if selected:
                self.bundle.activate(role)
                values = updater.update(selected)
                result.update({role + '/' + k: v for k, v in values.items()})
                result['optimizer_updates'] += 1
        return result

    def save(self, path, state):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.bundle.offload_all()
        for role, updater in self.bundle.updaters.items():
            updater.save(path / role, state)
        write_json(path / 'state.json', state)
        write_json(path / 'COMPLETE.json', {'separate_parameters': True})

    def restore(self, path):
        states = [updater.restore(Path(path) / role) for role, updater in self.bundle.updaters.items()]
        if not states or any(s != states[0] for s in states):
            raise ValueError('Separate-role checkpoint cursors differ')
        return states[0]

    def export(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for role, policy in self.bundle.policies.items():
            policy.save_model(path / role)
        write_json(path / 'roles.json', {role: role for role in self.bundle.policies})


def create(config, training=False, construction=None, retrieval=None):
    from models.hf import HFPolicy
    from training.native import NativeUpdater
    cfg = copy.deepcopy(config)
    cfg['model'].update(device='cpu', cpu_dtype=cfg['model']['dtype'])
    if construction or retrieval:
        builder = HFPolicy(construction or cfg['models']['base'], cfg)
        retriever = HFPolicy(retrieval or cfg['models']['base'], cfg)
        policies = {'B': builder, 'U': builder, 'R': retriever}
    else:
        policies = {role: HFPolicy(cfg['models']['base'], cfg) for role in ('B', 'U', 'R')}
    updaters = {role: NativeUpdater(policy, config) for role, policy in policies.items()} if training else {}
    bundle = RoleBundle(policies, config['model']['device'], updaters)
    return bundle, SeparateUpdater(bundle) if training else None
