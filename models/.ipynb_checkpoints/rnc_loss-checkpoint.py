import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import logger

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        # features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        # labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
    
# def ConR(features, targets, preds, w=1, weights=1, t=0.07, e=0.01):
    
#     if weights is None:
#         weights= 1
    
#     q = torch.nn.functional.normalize(features, dim=1)
#     k = torch.nn.functional.normalize(features, dim=1)

#     l_k = targets.flatten()[None, :]
#     l_q = targets

#     p_k = preds.flatten()[None, :]
#     p_q = preds

#     l_dist = torch.abs(l_q - l_k)
#     p_dist = torch.abs(p_q - p_k)
#     # print("l_dist: ", l_dist)

#     pos_i = l_dist.le(w)
#     neg_i = ((~ (l_dist.le(w))) * (p_dist.le(w)))
#     # print("neg_i: ", neg_i)

#     for i in range(pos_i.shape[0]):
#         pos_i[i][i] = 0

#     prod = torch.einsum("nc,kc->nk", [q, k]) / t
#     pos = prod * pos_i
#     neg = prod * neg_i

#     pushing_w = weights * torch.exp(l_dist * e)
#     # print("pushing_w: ", pushing_w)
#     neg_exp_dot = (pushing_w * (torch.exp(neg)) * neg_i).sum(1)
#     # print("neg_exp_dot: ", neg_exp_dot)
#     # For each query sample, if there is no negative pair, zero-out the loss.
#     no_neg_flag = (neg_i).sum(1).bool()
#     # print("no_neg_flag: ", no_neg_flag)

#     # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
#     denom = (pos_i.sum(1) + 1e-6)
#     # print("denom: ", denom)

#     loss = ((-torch.log(
#         torch.div(torch.exp(pos), (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1))) * (
#                  pos_i)).sum(1) / denom)
#     # print("loss: ", loss)
#     # print("loss: ",loss)
#     # print("no_neg_flag: ", no_neg_flag)
#     loss = (weights*(loss * no_neg_flag).unsqueeze(-1)).mean()
#     # print("loss 1: ", loss)

#     return loss
# def ConR(feature,depth,output,weights=1,w=0.1,t=0.1,e=0.01): # w=0.2,t=0.07,e=0.2
def ConR(feature,depth,output,weights=1,w=0.2,t=0.07,e=0.2):
    # logger.info(f"Using w: {w}")
    
    k = feature.reshape([feature.shape[0],-1])
    q = feature.reshape([feature.shape[0],-1])

    
    depth = depth.reshape(depth.shape[0],-1)
    l_k = torch.mean(depth,dim=1).unsqueeze(-1)
    l_q = torch.mean(depth,dim=1).unsqueeze(-1)

    output = output.reshape(output.shape[0],-1)
    p_k = torch.mean(output,dim=1).unsqueeze(-1)
    p_q = torch.mean(output,dim=1).unsqueeze(-1)
    
    
    
    
    l_dist = torch.abs(l_q -l_k.T)
    p_dist = torch.abs(p_q -p_k.T)

    


    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)
  
    Temp = 0.07
   
    # dot product of anchor with positives. Positives are keys with similar label
    pos_i = l_dist.le(w)
    neg_i = ((~ (l_dist.le(w)))*(p_dist.le(w)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0
    
    prod = torch.einsum("nc,kc->nk", [q, k])/t
    pos = prod * pos_i
    neg = prod * neg_i


    
    #  Pushing weight 
    if isinstance(weights, torch.Tensor):
        weights = torch.mean(weights.reshape(weights.shape[0],-1),dim=1).unsqueeze(-1)
    pushing_w = l_dist*weights*e

    

    # Sum exp of negative dot products
    neg_exp_dot=(pushing_w*(torch.exp(neg))*(neg_i)).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom=l_dist.le(w).sum(1)

    loss = ((-torch.log(torch.div(torch.exp(pos),(torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)
    
    
    
    
    loss = ((loss*no_neg_flag).unsqueeze(-1)).mean()
    
    

    return loss