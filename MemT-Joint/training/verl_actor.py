"""Node-normalized MoT objective on the pinned Tree-GRPO FSDP workers."""
import torch

from verl.workers.actor.dp_actor import DataParallelPPOActor
from methods.joint.loss import joint_loss
from methods.retrieval.loss import packed_response_log_probs


class MoTActor(DataParallelPPOActor):
    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        if not self.config.get('response_projection', False):
            return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)
        assert self.use_remove_padding and not self.use_ulysses_sp
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            log_probs = packed_response_log_probs(self.actor_module, micro_batch, temperature)
        # Neither the joint objective nor compute_log_prob consumes entropy.
        return torch.zeros_like(log_probs), log_probs

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
            loss, values = joint_loss(current, micro["old_log_probs"], micro["ref_log_prob"],
                                    micro["advantages"], micro["loss_mask"], micro["node_weight"],
                                    self.config.clip_ratio, self.config.kl_loss_coef,
                                    self.config.get('loss_variant', 'step'))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MoT policy loss")
            loss.backward()
            metrics["actor/loss"] += loss.detach().item()
            metrics["actor/kl"] += values["kl"]
        if self.config.get('defer_optimizer_load', False):
            from verl.utils.fsdp_utils import load_fsdp_optimizer
            load_fsdp_optimizer(self.actor_optimizer, torch.cuda.current_device())
        probes = [(p, p.detach().reshape(-1)[:4096].float().cpu().clone()) for p in self.actor_module.parameters()
                  if p.numel()] if self.config.get('parameter_probe', False) else []
        norm = self._optimizer_step()
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite MoT gradient")
        self.actor_optimizer.zero_grad()
        metrics["actor/grad_norm"] = norm.item()
        if probes:
            deltas = [p.detach().reshape(-1)[:4096].float().cpu() - before for p, before in probes]
            values = torch.tensor([sum(float(d.square().sum()) for d in deltas),
                                   sum(int(d.count_nonzero()) for d in deltas),
                                   sum(d.numel() for d in deltas)], device='cuda', dtype=torch.float64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(values)
            metrics.update({'actor/parameter_probe_delta_l2': values[0].sqrt().item(),
                            'actor/parameter_probe_changed': values[1].item(),
                            'actor/parameter_probe_size': values[2].item()})
        # Worker-group collection retains rank zero's metadata; report the global objective.
        if torch.distributed.is_initialized():
            totals = torch.tensor([metrics['actor/loss'], metrics['actor/kl']], device='cuda', dtype=torch.float32)
            torch.distributed.all_reduce(totals)
            totals /= torch.distributed.get_world_size()
            metrics['actor/loss'], metrics['actor/kl'] = totals.tolist()
        return metrics
