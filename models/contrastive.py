import torch

def CT_Regress(feature,depth,output,weights=None,w=0.2,t=0.07,e=0.01): 

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
  
    # Temp = 0.07
   
    # dot product of anchor with positives. Positives are keys with similar label
    pos_i = l_dist.le(w)
    neg_i = ((~ (l_dist.le(w)))*(p_dist.le(w)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0
    
    prod = torch.einsum("nc,kc->nk", [q, k])/t
    pos = prod * pos_i
    neg = prod * neg_i

    #  Pushing weight 
    if weights is None:
        weights = torch.ones_like(l_dist)
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


def CT_Single(feature,depth,output,weights=torch.tensor([1]),w=0.2,t=0.07,e=0.2, lamda=1):
    k = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)
    q = feature.reshape([feature.shape[0],-1]) # (batch_size, feature_dim)
    
    try:
        depth = depth.reshape(depth.shape[0],1) # target (batch_size, target_dim)
    except:
        depth = depth.reshape(depth.shape[0],-1)

    l_k = torch.tensor(depth)
    l_q = torch.tensor(depth)
    
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

def CT_Multi(feature,depth,output,weights=None,w=0.2,t=0.07,e=0.2, coef= 1):
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

    pos = prod * pos_i  
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