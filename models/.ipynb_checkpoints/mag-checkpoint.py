import torch
import torch.nn as nn
import torch.nn.functional as F
from .test_module import BertCrossEncoder, BertSelfEncoder
import pickle
from ..utils import logger
device= "cuda:0" if torch.cuda.is_available() else "cpu"


def pad_embedd(x, y):
    max_length = max(x.size(1), y.size(1))
    pad_size_x = max_length - x.size(1)
    pad_size_y = max_length - y.size(1)
    padded_tensor_x = F.pad(x, (0, 0, 0, pad_size_x))
    padded_tensor_y = F.pad(y, (0, 0, 0, pad_size_y))
    padded_tensor_x = padded_tensor_x.to(x.dtype)
    padded_tensor_y = padded_tensor_y.to(y.dtype)    
    assert len(x.size()) == len(padded_tensor_x.size())
    assert len(y.size()) == len(padded_tensor_y.size())
    return padded_tensor_x, padded_tensor_y
def pad_mask(x, y):
    max_length = max(x.size(1), y.size(1))
    pad_size_x = max_length - x.size(1)
    pad_size_y = max_length - y.size(1)
    padded_mask_x = F.pad(x, (0, pad_size_x))
    padded_mask_y = F.pad(y, (0, pad_size_y))
    padded_mask_x = padded_mask_x.to(x.dtype)
    padded_mask_y = padded_mask_y.to(y.dtype) 
    assert len(x.size()) == len(padded_mask_x.size())
    assert len(y.size()) == len(padded_mask_y.size())
    return padded_mask_x, padded_mask_y
class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()

        self.W_hv = nn.Linear(2*hidden_size, hidden_size)
        self.W_ha = nn.Linear(2*hidden_size, hidden_size)

        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.beta_shift = beta_shift

        self.LayerNorm_v1 = nn.LayerNorm(hidden_size)
        self.LayerNorm_v2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        logger.info("MAG model initialized, beta_shift: {}".format(beta_shift))

    def forward(self, text_embedding, text_embedding_v1, visual, visual_v1):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding_v1), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((text_embedding, visual_v1), dim=-1)))

        h_m = weight_v * self.W_v(text_embedding_v1) + weight_a * self.W_a(visual_v1)

        em_norm = (text_embedding_v1 + visual_v1).norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output_v1 = self.dropout(
            self.LayerNorm_v1(acoustic_vis_embedding + text_embedding_v1)
        )
        embedding_output_v2= self.dropout(
            self.LayerNorm_v2(acoustic_vis_embedding + visual_v1)
        )

        return embedding_output_v1, embedding_output_v2
    
class MAG_1(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG_1, self).__init__()

        self.W_hv = nn.Linear(2*hidden_size, hidden_size)
        self.W_ha = nn.Linear(2*hidden_size, hidden_size)

        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.beta_shift = beta_shift

        self.LayerNorm_v1 = nn.LayerNorm(hidden_size)
        self.LayerNorm_v2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        logger.info("MAG model initialized, beta_shift: {}".format(beta_shift))

    def forward(self, text_embedding, text_embedding_v1, visual, visual_v1):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding_v1), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((text_embedding, visual_v1), dim=-1)))

        h_m = weight_v * self.W_v(text_embedding_v1) + weight_a * self.W_a(visual_v1)

        em_norm = (text_embedding_v1 + visual_v1).norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output_v1 = self.dropout(
            self.LayerNorm_v1(acoustic_vis_embedding + text_embedding_v1 + visual_v1)
        )

        return embedding_output_v1

class MAG_(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG_, self).__init__()

        self.W_hv = nn.Linear(2*hidden_size, hidden_size)

        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.beta_shift = beta_shift

        self.LayerNorm_v1 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_embedding, visual):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual)

        em_norm = (text_embedding).norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output_v1 = self.dropout(
            self.LayerNorm_v1(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output_v1
    
class MAG_2(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG_2, self).__init__()

        self.W_hv = nn.Linear(2*hidden_size, hidden_size)
        self.W_ha = nn.Linear(2*hidden_size, hidden_size)

        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.beta_shift = beta_shift

        self.LayerNorm_v1 = nn.LayerNorm(hidden_size)
        self.LayerNorm_v2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_embedding, visual, visual_v1):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((text_embedding, visual), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((text_embedding, visual_v1), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(visual_v1)

        em_norm = (text_embedding).norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm_v2(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output

class MAGCrossEncoder(nn.Module): #0.2701
    def __init__(self, config):
        super(MAGCrossEncoder, self).__init__()
        self.txt2img_attention = BertCrossEncoder(config, 1)
        self.img2txt_attention = BertCrossEncoder(config, 1)
        self.mag = MAG(config.hidden_size, config.beta_shift, config.hidden_dropout_prob)
        self.attention_v1 = BertSelfEncoder(config)
        self.attention_v2 = BertSelfEncoder(config)

    def forward(self, text_embedding, visual_embedding, text_attention_mask, visual_attention_mask, fuse_type='mean'):
        # print(text_embedding.shape, visual_embedding.shape, text_attention_mask.shape, visual_attention_mask.shape)
        # padding mask & embedding
        text_embedding, visual_embedding = pad_embedd(text_embedding, visual_embedding)
        text_attention_mask, visual_attention_mask = pad_mask(text_attention_mask, visual_attention_mask)
        # create mask for text and visual embeddings

        extended_txt_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        extended_vis_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_vis_mask = extended_vis_mask.to(dtype=next(self.parameters()).dtype)
        extended_vis_mask = (1.0 - extended_vis_mask) * -10000.0

        # cross attention

        text_embedding_v1 = self.txt2img_attention(text_embedding, visual_embedding, extended_vis_mask)[-1]
        visual_embedding_v1 = self.img2txt_attention(visual_embedding, text_embedding, extended_txt_mask)[-1]
        # print(text_embedding_v1.shape, visual_embedding_v1.shape)
        # text_embedding_v2 = self.mag(text_embedding, text_embedding_v1, visual_embedding, visual_embedding_v1)[-1]
        # visual_embedding_v2 = self.mag(visual_embedding, visual_embedding_v1, text_embedding, text_embedding_v1)[-1]
        text_embedding_v2, visual_embedding_v2 = self.mag(text_embedding, text_embedding_v1, visual_embedding, visual_embedding_v1)
        # print(text_embedding_v2.shape, visual_embedding_v2.shape)
        text_embedding_v2 = self.attention_v1(text_embedding_v2, extended_txt_mask)[-1]
        visual_embedding_v2 = self.attention_v2(visual_embedding_v2, extended_vis_mask)[-1]
        # print(text_embedding_v2.shape, visual_embedding_v2.shape)

        # with open('embedding.pkl', 'wb') as f:
        #     pickle.dump([text_embedding, visual_embedding, text_embedding_v1, visual_embedding_v2, text_embedding_v2, visual_embedding_v2], f)

        if fuse_type=='mean':
            # print(text_embedding_v2.shape, visual_embedding_v2.shape, text_attention_mask.shape, visual_attention_mask.shape)
            text_embedding_v2[~text_attention_mask]=0
            
            visual_embedding_v2[~visual_attention_mask]=0

            final_output = torch.cat((text_embedding_v2, visual_embedding_v2), dim=1)

            classification_feats_pooled = final_output.sum(dim=1)/(text_attention_mask.sum(dim=1).view(-1, 1)+visual_attention_mask.sum(dim=1).view(-1, 1))
        

        return classification_feats_pooled
    
class CrossEncoder(nn.Module): #0.27..
    def __init__(self, config):
        super(CrossEncoder, self).__init__()
        self.txt2img_attention = BertCrossEncoder(config, 1)
        self.img2txt_attention = BertCrossEncoder(config, 1)

        self.attention_v1 = BertCrossEncoder(config, 1)
        self.attention_v2 = BertCrossEncoder(config, 1)

    def forward(self, text_embedding, visual_embedding, text_attention_mask, visual_attention_mask, fuse_type='mean'):
        # print(text_embedding.shape, visual_embedding.shape, text_attention_mask.shape, visual_attention_mask.shape)
        # padding mask & embedding
        # text_embedding, visual_embedding = pad_embedd(text_embedding, visual_embedding)
        # text_attention_mask, visual_attention_mask = pad_mask(text_attention_mask, visual_attention_mask)
        # create mask for text and visual embeddings

        extended_txt_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        extended_vis_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_vis_mask = extended_vis_mask.to(dtype=next(self.parameters()).dtype)
        extended_vis_mask = (1.0 - extended_vis_mask) * -10000.0

        # cross attention

        text_embedding_v1 = self.txt2img_attention(text_embedding, visual_embedding, extended_vis_mask)[-1]
        visual_embedding_v1 = self.img2txt_attention(visual_embedding, text_embedding, extended_txt_mask)[-1]

        text_embedding_v3 = self.attention_v1(text_embedding_v1, visual_embedding_v1, extended_vis_mask)[-1]
        visual_embedding_v3 = self.attention_v2(visual_embedding_v1, text_embedding_v1, extended_txt_mask)[-1]


        # with open('embedding.pkl', 'wb') as f:
        #     pickle.dump([text_embedding, visual_embedding, text_embedding_v1, visual_embedding_v2, text_embedding_v2, visual_embedding_v2], f)

        if fuse_type=='mean':
            # print(text_embedding_v2.shape, visual_embedding_v2.shape, text_attention_mask.shape, visual_attention_mask.shape)
            text_embedding_v3[~text_attention_mask]=0
            
            visual_embedding_v3[~visual_attention_mask]=0

            final_output = torch.cat((text_embedding_v3, visual_embedding_v3), dim=1)

            classification_feats_pooled = final_output.sum(dim=1)/(text_attention_mask.sum(dim=1).view(-1, 1)+visual_attention_mask.sum(dim=1).view(-1, 1))
        

        return classification_feats_pooled
class MAGCrossEncoder_1(nn.Module): #0.2701
    def __init__(self, config):
        super(MAGCrossEncoder_1, self).__init__()
        self.txt2img_attention = BertCrossEncoder(config, 1)
        self.img2txt_attention = BertCrossEncoder(config, 1)
        self.mag = MAG(config.hidden_size, config.beta_shift, config.hidden_dropout_prob)
        self.attention_v1 = BertCrossEncoder(config, 1)
        self.attention_v2 = BertCrossEncoder(config, 1)

    def forward(self, text_embedding, visual_embedding, text_attention_mask, visual_attention_mask, fuse_type='mean'):
        # print(text_embedding.shape, visual_embedding.shape, text_attention_mask.shape, visual_attention_mask.shape)
        # padding mask & embedding
        text_embedding, visual_embedding = pad_embedd(text_embedding, visual_embedding)
        text_attention_mask, visual_attention_mask = pad_mask(text_attention_mask, visual_attention_mask)
        # create mask for text and visual embeddings

        extended_txt_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        extended_vis_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_vis_mask = extended_vis_mask.to(dtype=next(self.parameters()).dtype)
        extended_vis_mask = (1.0 - extended_vis_mask) * -10000.0

        # cross attention

        text_embedding_v1 = self.txt2img_attention(text_embedding, visual_embedding, extended_vis_mask)[-1]
        visual_embedding_v1 = self.img2txt_attention(visual_embedding, text_embedding, extended_txt_mask)[-1]
        text_embedding_v2, visual_embedding_v2 = self.mag(text_embedding, text_embedding_v1, visual_embedding, visual_embedding_v1)

        text_embedding_v3 = self.attention_v1(text_embedding_v2, visual_embedding_v2, extended_vis_mask)[-1]
        visual_embedding_v3 = self.attention_v2(visual_embedding_v2, text_embedding_v2, extended_txt_mask)[-1]


        # with open('embedding.pkl', 'wb') as f:
        #     pickle.dump([text_embedding, visual_embedding, text_embedding_v1, visual_embedding_v2, text_embedding_v2, visual_embedding_v2], f)

        if fuse_type=='mean':
            # print(text_embedding_v2.shape, visual_embedding_v2.shape, text_attention_mask.shape, visual_attention_mask.shape)
            text_embedding_v3[~text_attention_mask]=0
            
            visual_embedding_v3[~visual_attention_mask]=0

            final_output = torch.cat((text_embedding_v3, visual_embedding_v3), dim=1)

            classification_feats_pooled = final_output.sum(dim=1)/(text_attention_mask.sum(dim=1).view(-1, 1)+visual_attention_mask.sum(dim=1).view(-1, 1))
        

        return classification_feats_pooled
    
class MAGCrossEncoder_(nn.Module): # 0.2713
    def __init__(self, config):
        super(MAGCrossEncoder_, self).__init__()
        self.txt2img_attention = BertCrossEncoder(config, 1)
        self.img2txt_attention = BertCrossEncoder(config, 1)
        self.mag_v1 = MAG_(config.hidden_size, config.beta_shift, config.hidden_dropout_prob)
        self.mag_v2 = MAG_(config.hidden_size, config.beta_shift, config.hidden_dropout_prob)
        self.attention_v1 = BertSelfEncoder(config)
        self.attention_v2 = BertSelfEncoder(config)

    def forward(self, text_embedding, visual_embedding, text_attention_mask, visual_attention_mask, fuse_type='mean'):
        
        # padding mask & embedding
        text_embedding, visual_embedding = pad_embedd(text_embedding, visual_embedding)
        text_attention_mask, visual_attention_mask = pad_mask(text_attention_mask, visual_attention_mask)
        # create mask for text and visual embeddings

        extended_txt_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        extended_vis_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_vis_mask = extended_vis_mask.to(dtype=next(self.parameters()).dtype)
        extended_vis_mask = (1.0 - extended_vis_mask) * -10000.0

        # cross attention

        text_embedding_v1 = self.txt2img_attention(text_embedding, visual_embedding, extended_vis_mask)[-1]
        visual_embedding_v1 = self.img2txt_attention(visual_embedding, text_embedding, extended_txt_mask)[-1]

        text_embedding_v2 = self.mag_v1(text_embedding_v1, visual_embedding_v1)
        visual_embedding_v2 = self.mag_v2(visual_embedding_v1, text_embedding_v1)

        text_embedding_v2 = self.attention_v1(text_embedding_v2, extended_txt_mask)[-1]
        visual_embedding_v2 = self.attention_v2(visual_embedding_v2, extended_vis_mask)[-1]

        if fuse_type=='mean':
            text_embedding_v2[~text_attention_mask]=0
            visual_embedding_v2[~visual_attention_mask]=0
            final_output = torch.cat((text_embedding_v2, visual_embedding_v2), dim=1)
            classification_feats_pooled = final_output.sum(dim=1)/(text_attention_mask.sum(dim=1).view(-1, 1)+visual_attention_mask.sum(dim=1).view(-1, 1))
        

        return classification_feats_pooled



class MAGCrossEncoder_2(nn.Module): # .2710
    def __init__(self, config):
        super(MAGCrossEncoder_2, self).__init__()
        self.txt2img_attention = BertCrossEncoder(config, 1)
        self.img2txt_attention = BertCrossEncoder(config, 1)
        self.mag_v1 = MAG_2(config.hidden_size, config.beta_shift, config.hidden_dropout_prob)
        self.mag_v2 = MAG_2(config.hidden_size, config.beta_shift, config.hidden_dropout_prob)
        self.attention_v1 = BertSelfEncoder(config)
        self.attention_v2 = BertSelfEncoder(config)

    def forward(self, text_embedding, visual_embedding, text_attention_mask, visual_attention_mask, fuse_type='mean'):
        
        # padding mask & embedding
        text_embedding, visual_embedding = pad_embedd(text_embedding, visual_embedding)
        text_attention_mask, visual_attention_mask = pad_mask(text_attention_mask, visual_attention_mask)
        # create mask for text and visual embeddings

        extended_txt_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        extended_vis_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_vis_mask = extended_vis_mask.to(dtype=next(self.parameters()).dtype)
        extended_vis_mask = (1.0 - extended_vis_mask) * -10000.0

        # cross attention

        text_embedding_v1 = self.txt2img_attention(text_embedding, visual_embedding, extended_vis_mask)[-1]
        visual_embedding_v1 = self.img2txt_attention(visual_embedding, text_embedding, extended_txt_mask)[-1]

        text_embedding_v2 = self.mag_v1(text_embedding_v1, visual_embedding, visual_embedding_v1)
        visual_embedding_v2 = self.mag_v2(visual_embedding_v1, text_embedding, text_embedding_v1)

        text_embedding_v2 = self.attention_v1(text_embedding_v2, extended_txt_mask)[-1]
        visual_embedding_v2 = self.attention_v2(visual_embedding_v2, extended_vis_mask)[-1]

        if fuse_type=='mean':
            text_embedding_v2[~text_attention_mask]=0
            visual_embedding_v2[~visual_attention_mask]=0
            final_output = torch.cat((text_embedding_v2, visual_embedding_v2), dim=1)
            classification_feats_pooled = final_output.sum(dim=1)/(text_attention_mask.sum(dim=1).view(-1, 1)+visual_attention_mask.sum(dim=1).view(-1, 1))
        

        return classification_feats_pooled
    
class MAGCrossEncoder_3(nn.Module): # .2707
    def __init__(self, config):
        super(MAGCrossEncoder_3, self).__init__()
        self.txt2img_attention = BertCrossEncoder(config, 1)
        self.img2txt_attention = BertCrossEncoder(config, 1)
        self.mag_v1 = MAG_2(config.hidden_size, config.beta_shift, config.hidden_dropout_prob)
        self.mag_v2 = MAG_2(config.hidden_size, config.beta_shift, config.hidden_dropout_prob)
        self.attention_v1 = BertCrossEncoder(config, 1)
        self.attention_v2 = BertCrossEncoder(config, 1)

    def forward(self, text_embedding, visual_embedding, text_attention_mask, visual_attention_mask, fuse_type='mean'):
        
        # padding mask & embedding
        text_embedding, visual_embedding = pad_embedd(text_embedding, visual_embedding)
        text_attention_mask, visual_attention_mask = pad_mask(text_attention_mask, visual_attention_mask)
        # create mask for text and visual embeddings

        extended_txt_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        extended_vis_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_vis_mask = extended_vis_mask.to(dtype=next(self.parameters()).dtype)
        extended_vis_mask = (1.0 - extended_vis_mask) * -10000.0

        # cross attention

        text_embedding_v1 = self.txt2img_attention(text_embedding, visual_embedding, extended_vis_mask)[-1]
        visual_embedding_v1 = self.img2txt_attention(visual_embedding, text_embedding, extended_txt_mask)[-1]

        text_embedding_v2 = self.mag_v1(text_embedding_v1, visual_embedding, visual_embedding_v1)
        visual_embedding_v2 = self.mag_v2(visual_embedding_v1, text_embedding, text_embedding_v1)

        text_embedding_v3 = self.attention_v1(text_embedding_v2, visual_embedding_v2, extended_vis_mask)[-1]
        visual_embedding_v3 = self.attention_v2(visual_embedding_v2, text_embedding_v2, extended_txt_mask)[-1]

        if fuse_type=='mean':
            text_embedding_v3[~text_attention_mask]=0
            visual_embedding_v3[~visual_attention_mask]=0
            final_output = torch.cat((text_embedding_v2, visual_embedding_v2), dim=1)
            classification_feats_pooled = final_output.sum(dim=1)/(text_attention_mask.sum(dim=1).view(-1, 1)+visual_attention_mask.sum(dim=1).view(-1, 1))
        

        return classification_feats_pooled
