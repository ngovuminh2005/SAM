"""Scores increase toward OOD; FPR95 uses ID as the positive class."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def ood_scores(logits, methods=('msp', 'energy'), temperature=1.):
    if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError('OOD scoring requires finite [batch, classes] logits')
    output = {}
    if 'msp' in methods:
        output['msp'] = 1 - logits.float().softmax(-1).max(-1).values
    if 'energy' in methods:
        output['energy'] = -temperature * torch.logsumexp(logits.float()/temperature, dim=-1)
    return output


def detection_metrics(id_scores, out_scores):
    inside, outside = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (id_scores, out_scores))
    if not len(inside) or not len(outside) or not np.isfinite(inside).all() or not np.isfinite(outside).all():
        raise ValueError('Detection scores must be nonempty and finite')
    scores = np.concatenate((inside, outside))
    labels_out = np.concatenate((np.zeros(len(inside)), np.ones(len(outside))))
    fpr, tpr, _ = roc_curve(1-labels_out, -scores, drop_intermediate=False)
    return {'auroc': float(roc_auc_score(labels_out, scores)),
            'aupr_out': float(average_precision_score(labels_out, scores)),
            'aupr_in': float(average_precision_score(1-labels_out, -scores)),
            'fpr95': float(fpr[tpr >= .95].min())}
