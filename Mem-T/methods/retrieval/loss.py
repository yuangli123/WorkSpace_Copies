"""Shared token objective for the reference trainer and distributed verl actor."""
import torch


def mot_loss(log_probs, old_log_probs, ref_log_probs, advantages, mask, weights, clip=0.2, beta=0.001):
    ratio = torch.exp(log_probs - old_log_probs)
    advantage = advantages if advantages.ndim == 2 else advantages[:, None]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    difference = ref_log_probs - log_probs
    kl = torch.exp(difference) - difference - 1
    tokens = mask.sum(-1).clamp_min(1)
    per_node = ((-surrogate + beta * kl) * mask).sum(-1) / tokens
    loss = (per_node * weights).sum()
    return loss, {"kl": ((kl * mask).sum(-1) / tokens * weights).sum().detach().item(),
                  "clip_fraction": (((ratio - 1).abs() > clip).float() * mask).sum().detach().item() / mask.sum().clamp_min(1).item()}


def causal_log_probs(model, prompt_ids, response_ids, device, temperature=1.0):
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    # Compute only action-token distributions. Tool observations are entirely in the prompt.
    logits = output.logits[:, len(prompt_ids) - 1:-1].float() / temperature
    labels = ids[:, len(prompt_ids):]
    return torch.log_softmax(logits, -1).gather(-1, labels[..., None]).squeeze(-1)
