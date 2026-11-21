"""Shared token objective for the reference trainer and distributed verl actor."""
import torch
import inspect


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
    kwargs = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'use_cache': False}
    limited = 'logits_to_keep' in inspect.signature(model.forward).parameters
    if limited:
        # Prefix hidden states still participate in attention/backprop; unused prefix logits need not be materialized.
        kwargs['logits_to_keep'] = len(response_ids) + 1
    output = model(**kwargs)
    # Compute only action-token distributions. Tool observations are entirely in the prompt.
    logits = output.logits[:, :-1].float() / temperature if limited else output.logits[:, len(prompt_ids) - 1:-1].float() / temperature
    labels = ids[:, len(prompt_ids):]
    return torch.log_softmax(logits, -1).gather(-1, labels[..., None]).squeeze(-1)


def packed_response_log_probs(model, batch, temperature):
    """Qwen3/FlashAttention packed forward; project only action predecessors."""
    ids, mask, positions = batch['input_ids'], batch['attention_mask'].bool(), batch['position_ids']
    width = batch['responses'].shape[-1]
    lengths = mask[:, -width:].sum(-1).tolist()
    prompts = mask[:, :-width].sum(-1).tolist()
    indices, labels, offset = [], [], 0
    for i, (prompt, length) in enumerate(zip(prompts, lengths)):
        indices.append(torch.arange(offset+prompt-1, offset+prompt+length-1, device=ids.device))
        labels.append(batch['responses'][i, :length])
        offset += prompt + length
    selected = torch.cat(indices).long()
    output = model(input_ids=ids[mask][None], attention_mask=None,
                   position_ids=positions[mask][None], use_cache=False, logits_to_keep=selected)
    logits = output.logits.squeeze(0).float() / temperature
    values = torch.log_softmax(logits, -1).gather(-1, torch.cat(labels)[:, None]).squeeze(-1)
    # Retain the original response padding/mask layout without projecting padding logits.
    return torch.stack([torch.nn.functional.pad(v, (0, width-n)) for v, n in zip(values.split(lengths), lengths)])
