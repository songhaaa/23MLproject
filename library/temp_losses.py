import numpy as np
import torch
import torch.nn.functional as F
import ot

def loss_unl(backbone, input_w, input_s, args):
    feat_tu_con, tmp = backbone(torch.cat((input_w, input_s), dim=0))
    logits_tu, disc_tu = tmp
    feat_tu_w, feat_tu_s = feat_tu_con.chunk(2)
    logits_tu_w, logits_tu_s = logits_tu.chunk(2)

    # sample-wise consistency
    pseudo_label = torch.softmax(logits_tu_w.detach() * args.T, dim=1)
    max_probs, targets_u = torch.max(pseudo_label, dim=1)
    consis_mask = max_probs.ge(args.threshold).float()
    L_pl = (F.cross_entropy(logits_tu_s, targets_u, reduction='none') * consis_mask).mean()

    # class-wise consistency
    prob_tu_w = torch.softmax(logits_tu_w, dim=1)
    prob_tu_s = torch.softmax(logits_tu_s, dim=1)
    L_con_cls = contras_cls(prob_tu_w, prob_tu_s)

    return L_con_cls, L_pl, consis_mask

def contras_cls(p1, p2):
    N, C = p1.shape
    cov = p1.t() @ p2

    cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
    cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
    loss = 0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C \
        + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
    return loss
