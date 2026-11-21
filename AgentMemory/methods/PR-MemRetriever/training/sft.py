from __future__ import annotations

import math
import os
import random
from pathlib import Path

import torch

from experiments.io import write_json, get_budget


from models.hf import render_prompt

def sft_example(tokenizer, operation, max_length):
    prompt = tokenizer.encode(render_prompt(tokenizer, operation["messages"]), add_special_tokens=False)
    text = operation["response"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = operation["target_span"]
    response, labels = [], []
    for token, (left, right) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        if left >= end:
            break
        response.append(token)
        labels.append(token if right > start and left < end else -100)
    if not text[end:].strip() and tokenizer.eos_token_id is not None:
        response.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    ids = prompt + response
    labels = [-100] * len(prompt) + labels
    if len(ids) > max_length:
        first_target = next((i for i, v in enumerate(labels) if v != -100), len(labels))
        if len(ids) - first_target + 1 > max_length:
            raise ValueError("Selected action exceeds cutoff; refusing partial-action supervision")
        ids, labels = ids[-max_length:], labels[-max_length:]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("No supervised tokens after cutoff")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def train_sft(config, selected, model_path, output, resume=None):
    import hashlib
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
    cfg = config["construction"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Split by originating model completion, not atomic calls, to avoid train/valid duplicates.
    groups = {}
    for op in selected:
        key = (op.get('source_sample_id', op["sample_id"].rsplit('__r', 1)[0]), str(op["messages"]), op["response"])
        groups.setdefault(key, []).append(op)
    keys = list(groups)
    random.Random(config["seed"]).shuffle(keys)
    valid_count = max(1, math.ceil(len(keys) * cfg["validation_fraction"]))
    if len(keys) < 2:
        raise ValueError("SFT requires at least two distinct completions for held-out validation")
    valid_keys = set(keys[:valid_count])
    train, valid = [], []
    rejected = []
    for key, ops in groups.items():
        destination = valid if key in valid_keys else train
        for op in ops:
            try:
                destination.append(sft_example(tokenizer, op, cfg["max_length"]))
            except ValueError as error:
                rejected.append({'op_id': op.get('op_id'), 'reason': str(error)})
    if not train or not valid:
        raise ValueError('No complete supervised actions remain in train or validation')
    before_alignment = len(train)
    global_batch = cfg['per_device_batch_size'] * cfg['gradient_accumulation'] * int(os.getenv('WORLD_SIZE', '1'))
    if cfg.get('global_batch_size') is not None:
        if global_batch != cfg['global_batch_size']:
            raise ValueError(f'Expected global batch {cfg["global_batch_size"]}, got {global_batch}')
        train = train[:len(train) // global_batch * global_batch]
        if not train:
            raise ValueError('Insufficient distinct actions for one complete global batch')
    use_bf16 = config["model"]["dtype"] == "bfloat16" and torch.cuda.is_available()
    policy = cfg.get('checkpoint_policy', 'full')
    if policy not in ('full', 'export_only') or resume and policy != 'full':
        raise ValueError('Resume requires checkpoint_policy=full')
    weight_bytes = sum(p.stat().st_size for p in Path(model_path).glob('*.safetensors'))
    meter = get_budget()
    if meter:
        # AdamW master/first/second moment states plus two safe rotating saves.
        meter.check(weight_bytes * (16 if policy == 'full' else 2), 'SFT checkpoint/export peak admission')
    extra = {'save_strategy': 'no' if policy == 'export_only' else 'steps', 'save_total_limit': 2,
             'dataloader_num_workers': 0, 'dataloader_pin_memory': False,
             'dataloader_drop_last': bool(cfg.get('global_batch_size'))}
    deepspeed = cfg.get('deepspeed')
    if isinstance(deepspeed, str):
        deepspeed = json.loads(Path(deepspeed).read_text(encoding='utf-8'))
    if deepspeed and deepspeed.get('zero_optimization', {}).get('offload_optimizer') and 'optimizer' not in deepspeed:
        # ZeRO CPU offload rejects Trainer's client AdamW by default. Let DS
        # instantiate its installed CPUAdam with the Trainer's exact settings.
        deepspeed['optimizer'] = {'type': 'AdamW', 'params': {'lr': 'auto', 'betas': 'auto', 'eps': 'auto', 'weight_decay': 'auto'}}
    arguments = TrainingArguments(output_dir=str(output), max_steps=cfg["steps"],
                                  per_device_train_batch_size=cfg["per_device_batch_size"],
                                  per_device_eval_batch_size=1, gradient_accumulation_steps=cfg["gradient_accumulation"],
                                  learning_rate=cfg["learning_rate"], lr_scheduler_type="cosine",
                                  warmup_ratio=cfg["warmup_ratio"], bf16=use_bf16,
                                  gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                                  logging_steps=1, save_steps=cfg["save_every"], eval_strategy="steps",
                                  eval_steps=cfg["save_every"], report_to=[], seed=config["seed"],
                                  full_determinism=cfg.get('full_determinism', False),
                                  deepspeed=deepspeed, remove_unused_columns=False,
                                  use_cpu=config["model"].get("device") == "cpu", **extra)
    # Create the DeepSpeed integration before model loading, so ZeRO partitions
    # parameters at initialization instead of loading four complete 4B models.
    model_kwargs = {'torch_dtype': torch.bfloat16 if use_bf16 else torch.float32}
    if use_bf16 and cfg.get('attention_implementation'):
        model_kwargs['attn_implementation'] = cfg['attention_implementation']
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False

    def fingerprint(model):
        value = hashlib.sha256()
        for name, parameter in model.named_parameters():
            local = getattr(parameter, 'ds_tensor', parameter).detach().reshape(-1)
            value.update(name.encode())
            value.update(local[:1024].cpu().view(torch.uint8).numpy().tobytes())
        return value.hexdigest()

    def state_digest(value):
        digest = hashlib.sha256()
        def update(item):
            if isinstance(item, torch.Tensor):
                data = item.detach().cpu().contiguous()
                digest.update(str((data.dtype, tuple(data.shape))).encode())
                digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(item, dict):
                for key in sorted(item, key=str):
                    digest.update(str(key).encode())
                    update(item[key])
            elif isinstance(item, (list, tuple)):
                for child in item:
                    update(child)
            else:
                digest.update(repr(item).encode())
        update(value)
        return digest.hexdigest()

    class Audit(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            from .cpu_adam import normalize_bias_cache
            zero = getattr(trainer.model_wrapped, 'optimizer', None)
            self.cpu_adam_bias_cache_normalized = normalize_bias_cache(getattr(zero, 'optimizer', None))
            self.before = fingerprint(model)
            self.resumed_step = state.global_step
            self.applied_lr = None
        def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
            self.applied_lr = optimizer.param_groups[0]['lr']
            if cfg.get('audit_full_state'):
                engine = trainer.model_wrapped
                zero = engine.optimizer
                base_optimizer = zero.optimizer
                parts = list(zero.fp32_partitioned_groups_flat)
                moments = [base_optimizer.state[p] for group in base_optimizer.param_groups for p in group['params']]
                self.record = {'step_before_update': state.global_step,
                               'fp32_master_sha256': state_digest(parts),
                               'optimizer_sha256': state_digest(moments),
                               'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in base_optimizer.param_groups],
                               'bf16_parameter_sha256': state_digest([getattr(p, 'ds_tensor', p) for p in model.parameters()]),
                               'torch_rng_sha256': state_digest(torch.get_rng_state()),
                               'cuda_rng_sha256': state_digest(torch.cuda.get_rng_state()),
                               'scheduler': trainer.lr_scheduler.state_dict(), 'microbatches': []}
        def on_step_end(self, args, state, control, **kwargs):
            if cfg.get('audit_full_state'):
                write_json(Path(output) / 'update-audit' / f'rank-{args.process_index}-step-{state.global_step}.json', self.record)
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in (logs or {}).values()):
                raise FloatingPointError('Non-finite SFT telemetry')
            if state.is_world_process_zero:
                telemetry = dict(step=state.global_step, target=cfg['steps'],
                                 applied_learning_rate=self.applied_lr, metrics=logs)
                write_json(Path(output) / 'PROGRESS.json', telemetry)
                kind = 'evaluation' if 'eval_loss' in (logs or {}) else 'train'
                write_json(Path(output) / 'metrics' / f'{kind}-{state.global_step:06d}.json', telemetry)
        def on_train_end(self, args, state, control, model=None, **kwargs):
            after = fingerprint(model)
            write_json(Path(output) / f'parameter-proof-rank-{args.process_index}.json',
                       {'before': self.before, 'after': after, 'changed': self.before != after,
                        'cpu_adam_bias_cache_normalized': self.cpu_adam_bias_cache_normalized,
                        'resumed_step': self.resumed_step, 'completed_step': state.global_step})

    audit = Audit()
    class AuditedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if cfg.get('audit_full_state') and model.training:
                audit.record['microbatches'].append(state_digest({k: inputs[k] for k in ('input_ids', 'labels', 'attention_mask')}))
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    trainer = AuditedTrainer(model=model, args=arguments, train_dataset=train, eval_dataset=valid,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True), tokenizer=tokenizer,
                      callbacks=[audit])
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = trainer.evaluate()
    if meter:
        meter.check(weight_bytes * 2, 'final HF SFT export')
    trainer.save_model(str(Path(output) / "final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(Path(output) / "final")
        write_json(Path(output) / "sft_audit.json", {"train_rows": len(train), "valid_rows": len(valid),
                                                    "completion_groups": len(keys), "train_completion_groups": len(keys) - valid_count,
                                                    "valid_completion_groups": valid_count,
                                                    "train_target_tokens": sum(sum(v != -100 for v in r["labels"]) for r in train),
                                                    "rejected_actions": rejected, "checkpoint_policy": policy,
                                                    "global_batch_size": global_batch,
                                                    "train_rows_before_batch_alignment": before_alignment,
                                                    "epoch_tail_excluded": before_alignment - len(train),
                                                    "train_completion_ids": [str(k) for k in keys if k not in valid_keys],
                                                    "valid_completion_ids": [str(k) for k in keys if k in valid_keys],
                                                    "metrics": metrics, "training_metrics": result.metrics,
                                                    "effective_deepspeed": deepspeed,
                                                    "cpu_adam_bias_cache_normalized": audit.cpu_adam_bias_cache_normalized,
                                                    "step": trainer.state.global_step})
        write_json(Path(output) / 'SFT_COMPLETE.json', {'step': trainer.state.global_step,
                   'checkpoint_policy': policy, 'full_4b_resume_verified': False,
                   'global_batch_size': global_batch})
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return metrics
