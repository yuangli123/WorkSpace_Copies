"""Node-normalized MoT objective on the pinned Tree-GRPO FSDP workers."""
import torch

from verl.workers.actor.dp_actor import DataParallelPPOActor
from methods.retrieval.loss import mot_loss


class MoTActor(DataParallelPPOActor):
    def update_policy(self, data):
        # One optimizer update per query batch; variable node counts must not change this.
        self.actor_module.train()
        for module in self.actor_module.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs",
                "ref_log_prob", "advantages", "loss_mask", "node_weight"]
        batch = data.select(batch_keys=keys).batch
        self.actor_optimizer.zero_grad()
        metrics = {"actor/loss": 0.0, "actor/kl": 0.0,
                   "actor/update_lr": float(self.actor_optimizer.param_groups[0]["lr"])}
        for micro in batch.split(self.config.ppo_micro_batch_size):
            micro = micro.cuda()
            _, current = self._forward_micro_batch(micro, data.meta_info["temperature"])
            loss, values = mot_loss(current, micro["old_log_probs"], micro["ref_log_prob"],
                                    micro["advantages"], micro["loss_mask"], micro["node_weight"],
                                    self.config.clip_ratio, self.config.kl_loss_coef)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MoT policy loss")
            loss.backward()
            metrics["actor/loss"] += loss.detach().item()
            metrics["actor/kl"] += values["kl"]
        norm = self._optimizer_step()
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite MoT gradient")
        self.actor_optimizer.zero_grad()
        metrics["actor/grad_norm"] = norm.item()
        return metrics
