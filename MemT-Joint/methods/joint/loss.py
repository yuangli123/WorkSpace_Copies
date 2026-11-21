"""Length-normalized step ratio; observations and old/reference policies are detached."""
import torch


def joint_loss(log_probs, old_log_probs, ref_log_probs, advantages, mask, weights,
               clip=0.2, beta=0.001, variant='step'):
    old_log_probs, ref_log_probs = old_log_probs.detach(), ref_log_probs.detach()
    advantages = advantages.detach()
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    mask = mask.to(log_probs.dtype)
    lengths = mask.sum(-1).clamp_min(1)
    differences = torch.where(mask.bool(), log_probs - old_log_probs, torch.zeros_like(log_probs))
    if variant == 'step':
        ratio = torch.exp((differences * mask).sum(-1) / lengths)
        a = (advantages.expand_as(log_probs) * mask).sum(-1) / lengths
        surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
        clipped = (ratio - 1).abs() > clip
    elif variant == 'token':
        ratio = torch.exp(differences)
        surrogate = (torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages) * mask).sum(-1) / lengths
        clipped = (((ratio - 1).abs() > clip) * mask).sum(-1) / lengths
    else:
        raise ValueError('Unknown joint loss: ' + variant)
    delta = torch.where(mask.bool(), ref_log_probs - log_probs, torch.zeros_like(log_probs))
    kl = ((torch.exp(delta) - delta - 1) * mask).sum(-1) / lengths
    loss = ((-surrogate + beta * kl) * weights).sum()
    return loss, {'kl': (kl * weights).sum().detach().item(),
                  'clip_fraction': (clipped.to(log_probs.dtype) * weights).sum().detach().item()}
