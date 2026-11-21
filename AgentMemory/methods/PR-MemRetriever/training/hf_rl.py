from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import torch

from experiments.io import read_json, write_json
from methods.retrieval.loss import causal_log_probs, mot_loss

class HFUpdater:
    def __init__(self, policy, config, reference=None):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Reference RL backend is single-process; select verl for distributed RL")
        self.policy, self.config = policy, config
        self.reference = reference or copy.deepcopy(policy.model)
        self.ref_device = torch.device(config["model"].get("reference_device", "cpu"))
        self.reference.to(self.ref_device).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(policy.model.parameters(), lr=config["retrieval"]["learning_rate"], weight_decay=0.0)
        warmup = int(config["retrieval"]["steps"] * config["retrieval"]["warmup_ratio"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min((step + 1) / max(1, warmup), 1.0))
        if config["model"].get("gradient_checkpointing", True):
            policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def update(self, rows):
        temperature = self.config["model"]["temperature"]
        if temperature <= 0:
            raise ValueError("RL requires stochastic sampling temperature > 0")
        old, refs = [], []
        self.policy.model.eval()
        with torch.no_grad():
            for row in rows:
                old.append(causal_log_probs(self.policy.model, row["prompt_ids"], row["response_ids"], self.policy.device, temperature).cpu())
                refs.append(causal_log_probs(self.reference, row["prompt_ids"], row["response_ids"], self.ref_device, temperature).cpu())
        self.optimizer.zero_grad(set_to_none=True)
        self.policy.model.train()  # HF gradient checkpointing is inactive in eval mode.
        total_loss, total_kl = 0.0, 0.0
        # Each unique node appears once, regardless of the number of descendant paths.
        for row, previous, reference in zip(rows, old, refs):
            current = causal_log_probs(self.policy.model, row["prompt_ids"], row["response_ids"], self.policy.device, temperature)
            loss, metrics = mot_loss(current, previous.to(current.device), reference.to(current.device),
                                     current.new_tensor([row["advantage"]]), torch.ones_like(current),
                                     current.new_tensor([row["weight"]]), self.config["retrieval"]["clip"], self.config["retrieval"]["kl_beta"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite policy loss")
            loss.backward()
            total_loss += loss.detach().item()
            total_kl += metrics["kl"]
        norm = torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite policy gradient")
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": total_loss, "kl": total_kl, "grad_norm": norm.item(),
                "learning_rate": self.scheduler.get_last_lr()[0]}

    def save(self, path, state):
        path = Path(path)
        self.policy.save_model(path / "model")
        self.reference.save_pretrained(path / "reference", safe_serialization=True)
        torch.save({"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path / "training_state.pt")
        write_json(path / "state.json", state)
        (path / "COMPLETE").write_text("ok\n", encoding="utf-8")

    def restore(self, path):
        from transformers import AutoModelForCausalLM
        path = Path(path)
        if not (path / "COMPLETE").exists():
            raise ValueError("Incomplete checkpoint")
        saved = AutoModelForCausalLM.from_pretrained(path / "model")
        self.policy.model.load_state_dict(saved.state_dict())
        ref = AutoModelForCausalLM.from_pretrained(path / "reference")
        self.reference.load_state_dict(ref.state_dict())
        # Only load checkpoints produced by this run; pickle is not an untrusted interchange format.
        saved_state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(saved_state["optimizer"])
        self.scheduler.load_state_dict(saved_state["scheduler"])
        torch.set_rng_state(saved_state["torch_rng"])
        random.setstate(saved_state["python_rng"])
        if saved_state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved_state["cuda_rng"])
        return read_json(path / "state.json")


