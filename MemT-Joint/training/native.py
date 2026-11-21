from __future__ import annotations

import copy
import inspect
import math
import random
from pathlib import Path

import torch

from experiments.io import read_json, write_json, sha256
from methods.retrieval.loss import causal_log_probs
from methods.joint.loss import joint_loss


class NativeUpdater:
    def __init__(self, policy, config):
        self.policy, self.config = policy, config
        policy.training_updater = self
        self.reference = copy.deepcopy(policy.model).cpu().eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(policy.model.parameters(), lr=config['retrieval']['learning_rate'], weight_decay=0)
        warmup = int(config['retrieval']['steps'] * config['retrieval']['warmup_ratio'])
        total = config['retrieval']['steps']
        def schedule(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        if config['model'].get('gradient_checkpointing', True) and hasattr(policy.model, 'gradient_checkpointing_enable'):
            policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def move_optimizer(self, device):
        for values in self.optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value) and key != 'step':
                    values[key] = value.to(device)

    def update(self, rows):
        if not rows:
            raise ValueError('No trainable rows; use evaluate for a frozen baseline')
        model, temperature = self.policy.model, self.config['model']['temperature']
        common = self.config.get('protocol', {}).get('name') in ('memt-common-v1', 'memt-execution-v2')
        if common:
            self.policy.device = torch.device(self.config['model']['device'])
            model.to(self.policy.device)
            self.move_optimizer('cpu')
        model.eval()
        with torch.no_grad():
            for row in rows:
                old = causal_log_probs(model, row['prompt_ids'], row['response_ids'], self.policy.device, temperature).cpu()
                ref = causal_log_probs(self.reference, row['prompt_ids'], row['response_ids'], 'cpu', temperature).cpu()
                row['old_log_probs'], row['ref_log_probs'] = old[0].tolist(), ref[0].tolist()
        self.optimizer.zero_grad(set_to_none=True)
        model.train()
        metrics = {'loss': 0.0, 'kl': 0.0, 'train_rows': len(rows), 'learning_rate': self.scheduler.get_last_lr()[0]}
        for row in rows:
            current = causal_log_probs(model, row['prompt_ids'], row['response_ids'], self.policy.device, temperature)
            old = current.new_tensor([row['old_log_probs']])
            ref = current.new_tensor([row['ref_log_probs']])
            loss, values = joint_loss(current, old, ref, current.new_tensor([row['advantage']]),
                                      torch.ones_like(current), current.new_tensor([row['weight']]),
                                      self.config['retrieval']['clip'], self.config['retrieval']['kl_beta'],
                                      self.config['joint']['loss_variant'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite shared loss')
            loss.backward()
            metrics['loss'] += loss.detach().item()
            metrics['kl'] += values['kl']
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite shared gradient')
        if common:
            self.move_optimizer(self.policy.device)
        self.optimizer.step()
        self.scheduler.step()
        if common:
            self.move_optimizer('cpu')
            if getattr(self.policy, 'generation', None):
                self.policy.generation.invalidate()
        metrics['grad_norm'] = norm.item()
        return metrics

    def save(self, path, state):
        import numpy as np
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        payload = {'model': self.policy.model.state_dict(), 'reference': self.reference.state_dict(),
                   'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                   'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(payload, path / 'training.pt')
        write_json(path / 'state.json', state)
        write_json(path / 'COMPLETE.json', {'training_sha256': sha256(path / 'training.pt'), 'state_sha256': sha256(path / 'state.json')})

    def restore(self, path):
        import numpy as np
        path = Path(path)
        marker = read_json(path / 'COMPLETE.json')
        if marker != {'training_sha256': sha256(path / 'training.pt'), 'state_sha256': sha256(path / 'state.json')}:
            raise ValueError('Checkpoint hash mismatch')
        kwargs = {'map_location': 'cpu'}
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = False
        data = torch.load(path / 'training.pt', **kwargs)
        self.policy.model.load_state_dict(data['model'])
        if getattr(self.policy, 'generation', None):
            self.policy.generation.invalidate()
        self.reference.load_state_dict(data['reference'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.scheduler.load_state_dict(data['scheduler'])
        torch.set_rng_state(data['torch_rng'])
        random.setstate(data['python_rng'])
        np.random.set_state(data['numpy_rng'])
        if data['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(data['cuda_rng'])
        return read_json(path / 'state.json')

    def export(self, path):
        self.policy.save_model(path)
