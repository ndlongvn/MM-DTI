import torch
import torch.nn.functional as F
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InfoNCE(nn.Module):

    def __init__(self, bert_output_size, graph_ouput_size, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.orig_d_l = bert_output_size
        self.orig_d_av = graph_ouput_size
        self.d_l, self.d_av = 50, 50 ### 50, 50, 30, 30

        self.embed_dropout = 0.1
        self.training = True

        self.info_proj_query = nn.Sequential(nn.Linear(self.orig_d_l, self.orig_d_l), nn.GELU(), nn.Linear(self.orig_d_l, self.d_l))
        self.info_proj_positive = nn.Sequential(nn.Linear(self.orig_d_av, self.orig_d_av), nn.GELU(), nn.Linear(self.orig_d_av, self.d_av))

    def forward(self, query, positive_key, negative_keys=None):
        x_l_ = F.dropout(query, p=self.embed_dropout, training=self.training)
        x_av_ = positive_key

        # Project the textual/visual/audio features
        proj_x_l = x_l_ if self.orig_d_l == self.d_l else self.info_proj_query(x_l_)
        proj_x_av = x_av_ if self.orig_d_av == self.d_av else self.info_proj_positive(x_av_)


        proj_query = torch.mean(proj_x_l, dim=1)
        proj_positive = torch.mean(proj_x_av, dim=1)

        return info_nce(proj_query, proj_positive, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)
    


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    # query dim != positive_key dim
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return (F.cross_entropy(logits / temperature, labels, reduction=reduction) + F.cross_entropy(logits.T / temperature, labels, reduction=reduction))/2

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]