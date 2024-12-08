import torch
import torch.nn.functional as F



import copy
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def ConR_single(feature,depth,output,weights=torch.tensor([1]),w=0.2,t=0.07,e=0.2, lamda=5):
    k = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)
    q = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)
    
    try:
        depth = depth.reshape(depth.shape[0],1) # target (batch_size, target_dim)
    except:
        depth = depth.reshape(depth.shape[0],-1)
    # l_k = torch.mean(depth,dim=1).unsqueeze(-1) # (batch_size, 1)
    # l_q = torch.mean(depth,dim=1).unsqueeze(-1) # (batch_size, 1)
    l_k = torch.tensor(depth)
    l_q = torch.tensor(depth)

    output = output[:, 1:]  # prediction (batch_size, 1): get only the positive class probability
    threshold = 0.5
    output = (output > threshold).float()
    p_k = torch.mean(output,dim=1).unsqueeze(-1) # (batch_size, 1)
    p_q = torch.mean(output,dim=1).unsqueeze(-1) # (batch_size, 1)
    
    l_dist = torch.abs(l_q -l_k.T)


    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)

    # dot product of anchor with positives. Positives are keys with similar label
    pos_i = ((l_dist.eq(0)))

    neg_i = (~(l_dist.eq(0)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0
    
    prod = torch.einsum("nc,kc->nk", [q, k])/t # (batch_size, batch_size): dot product of query and key

    pos = prod * pos_i
    neg = prod * neg_i
    #  Pushing weight 
    # weights = torch.mean(weights.reshape(weights.shape[0],-1),dim=1).unsqueeze(-1)
    # weights =  torch.tensor([1])
    pushing_w = weights.to(depth.device)#l_dist*weights*e

    # Sum exp of negative dot products
    neg_exp_dot=(pushing_w*(torch.exp(neg))*(neg_i)).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()


    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom=pos_i.sum(1)

    # Avoid division by zero
    denom[denom==0]=1

    loss = ((-torch.log(torch.div(torch.exp(pos),(torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)

    loss = ((loss*no_neg_flag).unsqueeze(-1)).mean()
    
    return loss

import torch # sai: implement laij
def ConR_2(feature,depth,output,weights=torch.tensor([1]),w=0.2,t=0.07,e=0.2, lamda=0.3):

    def create_matrix(arr):
        n= len(arr)
        result= torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                result[i, j] = arr[(j+i)%n]
        for i in range(n):
            result[i, i]= 1
        return result
    
    k = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)
    q = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)

    
    depth = depth.reshape(depth.shape[0],1) # target (batch_size, target_dim)
    # l_k = torch.mean(depth,dim=1).unsqueeze(-1) # (batch_size, 1)
    # l_q = torch.mean(depth,dim=1).unsqueeze(-1) # (batch_size, 1)
    l_k = torch.tensor(depth)
    l_q = torch.tensor(depth)

    output = F.softmax(output, dim=-1)
    output = output[:, 1:]  # prediction (batch_size, 1): get only the positive class probability
    threshold = 0.5 #calculate_single_classification_threshold(depth, output, metrics_key=[f1_score, True, 'int'], step=20)
    output = (output > threshold).float()
    
    l_dist = torch.abs(l_q -l_k.T) # real label distance

    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)

    # dot product of anchor with positives. Positives are keys with similar label
    pos_i = ((l_dist.eq(0)))

    pos_pred= (depth == output)  # predict and real label are the same
    neg_pred= (depth != output)  # predict and real label are different

    weight_pos = neg_pred* torch.Tensor([lamda-1.0]).to(depth.device) + torch.Tensor([1.0]).to(depth.device)
    weight_pos = create_matrix(weight_pos).to(depth.device)

    neg_i = (~(l_dist.eq(0)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0
    
    prod = torch.einsum("nc,kc->nk", [q, k])/t # (batch_size, batch_size): dot product of query and key

    pos = prod * pos_i
    neg = prod * neg_i
    #  Pushing weight 
    # weights = torch.mean(weights.reshape(weights.shape[0],-1),dim=1).unsqueeze(-1)
    # weights =  torch.tensor([1])
    pushing_w = weights.to(depth.device)#l_dist*weights*e

    # Sum exp of negative dot products
    neg_exp_dot=(pushing_w*(torch.exp(neg))*(neg_i)).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()


    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom=pos_i.sum(1)

    # Avoid division by zero
    denom[denom==0]=1

    pos_exp_dot= torch.exp(pos)*weight_pos
    loss = ((-torch.log(torch.div(pos_exp_dot,(pos_exp_dot.sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)
    loss = ((loss*no_neg_flag).unsqueeze(-1)).mean()
    
    return loss


def ConR(feature,depth,output,weights=torch.tensor([1]),w=0.2,t=0.07,e=0.2):
    k = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)
    q = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)

    num_classes = depth.shape[1]
    
    depth = depth.reshape(depth.shape[0], num_classes) # target (batch_size, target_dim)
    # l_k = torch.mean(depth,dim=1).unsqueeze(-1) # (batch_size, 1)
    # l_q = torch.mean(depth,dim=1).unsqueeze(-1) # (batch_size, 1)
    l_k = torch.tensor(depth)
    l_q = torch.tensor(depth)

    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)

    loss_all = 0

    for class_idx in range(num_classes):

        l_dist = torch.abs(l_q[:, class_idx].unsqueeze(-1) -l_k[:, class_idx].unsqueeze(-1).T)


        # dot product of anchor with positives. Positives are keys with similar label
        pos_i = ((l_dist.eq(0)))

        neg_i = (~(l_dist.eq(0)))

        for i in range(pos_i.shape[0]):
            pos_i[i][i] = 0

        prod = torch.einsum("nc,kc->nk", [q, k])/t

        pos = prod * pos_i
        neg = prod * neg_i
        #  Pushing weight
        # weights = torch.mean(weights.reshape(weights.shape[0],-1),dim=1).unsqueeze(-1)
        pushing_w = weights.to(depth.device)#l_dist*weights*e
        
        # Sum exp of negative dot products
        neg_exp_dot=(pushing_w*(torch.exp(neg))*(neg_i)).sum(1)

        # For each query sample, if there is no negative pair, zero-out the loss.
        no_neg_flag = (neg_i).sum(1).bool()

        # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
        denom=pos_i.sum(1)

        # Avoid division by zero
        denom[denom==0]=1

        loss = ((-torch.log(torch.div(torch.exp(pos),(torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)
        # print(loss)
        loss = ((loss*no_neg_flag).unsqueeze(-1)).mean()

        loss_all += loss
    

    return loss_all 


import torch
import torch.nn.functional as F
def ConR_single_1(feature,depth,output,weights=torch.tensor([1]),w=0.2,t=0.07,e=0.2, lamda=0.5):
    
    k = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)
    q = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)

    
    depth = depth.reshape(depth.shape[0],1) # target (batch_size, target_dim)

    l_k = torch.tensor(depth)
    l_q = torch.tensor(depth)

    output = F.softmax(output, dim=-1)
    output = output[:, 1:]  # prediction (batch_size, 1): get only the positive class probability
    threshold = 0.5
    output = (output > threshold).float()
    
    l_dist = torch.abs(l_q -l_k.T) # real label distance

    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)

    # dot product of anchor with positives. Positives are keys with similar label
    pos_i = ((l_dist.eq(0)))
    neg_i = (~(l_dist.eq(0)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0

    pos_pred= (depth == output)  # predict and real label are the same
    neg_pred= (depth != output)  # predict and real label are different

    weight_pos = torch.ones_like(pos_i).float()
    for i in range(pos_i.shape[0]):
        for j in range(pos_i.shape[1]):
            if neg_pred[j]:
                weight_pos[i, j]= lamda
    
    prod = torch.einsum("nc,kc->nk", [q, k])/t # (batch_size, batch_size): dot product of query and key

    pos = prod * pos_i
    neg = prod * neg_i

    #  Pushing weight 
    # weights = torch.mean(weights.reshape(weights.shape[0],-1),dim=1).unsqueeze(-1)
    # weights =  torch.tensor([1])
    pushing_w = weights.to(depth.device)#l_dist*weights*e

    # Sum exp of negative dot products
    neg_exp_dot=(pushing_w*(torch.exp(neg))*(neg_i)).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()


    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom=pos_i.sum(1)

    # Avoid division by zero
    denom[denom==0]=1

    pos_exp = torch.exp(pos)
    pos_exp_dot= pos_exp*weight_pos
    loss = ((-torch.log(torch.div(pos_exp,(pos_exp_dot.sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)
    loss = ((loss*no_neg_flag).unsqueeze(-1)).mean()
    return loss

def ConR_Multi(feature,depth,output,weights=None,w=0.2,t=0.07,e=0.2, coef= 1, lamda=1):
    def compute_comparison_matrix(depth):
        # Initialize the comparison matrix
        comparison_matrix = torch.zeros((depth.shape[0], depth.shape[0]), dtype=torch.float32)
        num_classes = depth.shape[1]
        # Compare each pair of labels
        for i in range(depth.shape[0]):
            for j in range(depth.shape[0]):
                comparison_matrix[i, j] = torch.sum(depth[i] == depth[j])/num_classes

        return comparison_matrix
    k = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)
    q = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)

    depth = depth.reshape(depth.shape[0],-1) # target (batch_size, target_dim)

    l_dist = compute_comparison_matrix(depth).to(depth.device)

    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)

    # dot product of anchor with positives. Positives are keys with similar label
    threshold= coef/depth.shape[1]
    pos_i = ((l_dist.ge(threshold)))
    neg_i = (~(l_dist.ge(threshold)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0
    
    prod = torch.einsum("nc,kc->nk", [q, k])/t # (batch_size, batch_size): dot product of query and key

    pos = prod * pos_i * l_dist 
    neg = prod * neg_i
    #  Pushing weight 
    
    if weights is not None:
        pushing_w = weights.to(depth.device)

        # Sum exp of negative dot products
        neg_exp_dot=(pushing_w*(torch.exp(neg))*(neg_i)).sum(1)
    else:
        neg_exp_dot= (torch.exp(neg)*(neg_i)).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()


    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom=pos_i.sum(1)

    # Avoid division by zero
    denom[denom==0]=1

    loss = ((-torch.log(torch.div(torch.exp(pos),(torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)
    loss = ((loss*no_neg_flag).unsqueeze(-1)).mean()
    
    return loss