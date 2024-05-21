import torch
import torch.nn.functional as F
from functools import partial

def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def sce_loss(x, y, alpha=1):
    if x.numel() != 0:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
    else:
        loss = 0
    return loss

def setup_loss_fn(alpha_l):
    criterion = partial(sce_loss, alpha=alpha_l)
    return criterion

