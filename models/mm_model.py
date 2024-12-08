from __future__ import absolute_import, division, print_function
from ast import Not
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore.utils import get_activation_fn
from unicore.data import Dictionary
from unicore.models import BaseUnicoreModel
from unicore.modules import LayerNorm, init_bert_params
from models.transformers import TransformerEncoderWithPair
from models.mm_module import BertCrossEncoder
from utils import pad_1d_tokens, pad_2d, pad_coords
from models.infonce import InfoNCE
# from .conr import ConR
from models.fds import FDS
import argparse
import pathlib
import os
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, AutoConfig
# from transformers import AutoModel, AutoConfig
import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils import calibrate_mean_var, logger
from config import MODEL_CONFIG

BACKBONE = {
    'transformer': TransformerEncoderWithPair,
}

WEIGHT_DIR = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'weights')

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        """
        Initialize the classification head.

        :param input_dim: Dimension of input features.
        :param inner_dim: Dimension of the inner layer.
        :param num_classes: Number of classes for classification.
        :param activation_fn: Activation function name.
        :param pooler_dropout: Dropout rate for the pooling layer.
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        """
        Forward pass for the classification head.

        :param features: Input features for classification.

        :return: Output from the classification head.
        """
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NonLinearHead(nn.Module):
    """
    A neural network module used for simple classification tasks. It consists of a two-layered linear network 
    with a nonlinear activation function in between.

    Attributes:
        - linear1: The first linear layer.
        - linear2: The second linear layer that outputs to the desired dimensions.
        - activation_fn: The nonlinear activation function.
    """
    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        """
        Initializes the NonLinearHead module.

        :param input_dim: Dimension of the input features.
        :param out_dim: Dimension of the output.
        :param activation_fn: The activation function to use.
        :param hidden: Dimension of the hidden layer; defaults to the same as input_dim if not provided.
        """
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Forward pass of the NonLinearHead.

        :param x: Input tensor to the module.

        :return: Tensor after passing through the network.
        """
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
    
class GasModel(nn.Module):
    """
    Model for embedding gas attributes.
    """
    def __init__(self, gas_attr_input_dim, gas_dim, gas_max_count=500):
        """
        Initialize the GasModel.

        :param gas_attr_input_dim: Input dimension for gas attributes.
        :param gas_dim: Dimension for gas embeddings.
        :param gas_max_count: Maximum count for gas embedding.
        """
        super().__init__()
        self.gas_embed = nn.Embedding(gas_max_count, gas_dim)
        self.gas_attr_embed = NonLinearHead(gas_attr_input_dim, gas_dim, 'relu')

    def forward(self, gas, gas_attr):
        """
        Forward pass for the gas model.

        :param gas: Gas identifiers.
        :param gas_attr: Gas attributes.

        :return: Combined representation of gas and its attributes.
        """
        gas = gas.long()
        gas_attr = gas_attr.type_as(self.gas_attr_embed.linear1.weight)
        gas_embed = self.gas_embed(gas)  # shape of gas_embed is [batch_size, gas_dim]
        gas_attr_embed = self.gas_attr_embed(gas_attr)  # shape of gas_attr_embed is [batch_size, gas_dim]
        # gas_embed = torch.cat([gas_embed, gas_attr_embed], dim=-1)
        gas_repr = torch.concat([gas_embed, gas_attr_embed], dim=-1)
        return gas_repr

class EnvModel(nn.Module):
    """
    Model for environmental embeddings like pressure and temperature.
    """
    def __init__(self, hidden_dim, bins=32, min_max_key=None):
        """
        Initialize the EnvModel.

        :param hidden_dim: Dimension for the hidden layer.
        :param bins: Number of bins for embedding.
        :param min_max_key: Dictionary with min and max values for normalization.
        """
        super().__init__()
        self.project = NonLinearHead(2, hidden_dim, 'relu')
        self.bins = bins
        self.pressure_embed = nn.Embedding(bins, hidden_dim)
        self.temperature_embed = nn.Embedding(bins, hidden_dim)
        self.min_max_key = min_max_key
        
    def forward(self, pressure, temperature):
        """
        Forward pass for the environmental model.

        :param pressure: Pressure values.
        :param temperature: Temperature values.

        :return: Combined representation of environmental features.
        """
        pressure = pressure.type_as(self.project.linear1.weight)
        temperature = temperature.type_as(self.project.linear1.weight)
        pressure = torch.clamp(pressure, self.min_max_key['pressure'][0], self.min_max_key['pressure'][1])
        temperature = torch.clamp(temperature, self.min_max_key['temperature'][0], self.min_max_key['temperature'][1])
        pressure = (pressure - self.min_max_key['pressure'][0]) / (self.min_max_key['pressure'][1] - self.min_max_key['pressure'][0])
        temperature = (temperature - self.min_max_key['temperature'][0]) / (self.min_max_key['temperature'][1] - self.min_max_key['temperature'][0])
        # shapes of pressure and temperature both are [batch_size, ]
        env_project = torch.cat((pressure[:, None], temperature[:, None]), dim=-1)
        env_project = self.project(env_project)  # shape of env_project is [batch_size, env_dim]

        pressure_bin = torch.floor(pressure * self.bins).to(torch.long)
        temperature_bin = torch.floor(temperature * self.bins).to(torch.long)
        pressure_embed = self.pressure_embed(pressure_bin)  # shape of pressure_embed is [batch_size, env_dim]
        temperature_embed = self.temperature_embed(temperature_bin)  # shape of temperature_embed is [batch_size, env_dim]
        env_embed = torch.cat([pressure_embed, temperature_embed], dim=-1)

        env_repr = torch.cat([env_project, env_embed], dim=-1)

        return env_repr

@torch.jit.script
def gaussian(x, mean, std):
    """
    Gaussian function implemented for PyTorch tensors.

    :param x: The input tensor.
    :param mean: The mean for the Gaussian function.
    :param std: The standard deviation for the Gaussian function.

    :return: The output tensor after applying the Gaussian function.
    """
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
class GaussianLayer(nn.Module):
    """
    A neural network module implementing a Gaussian layer, useful in graph neural networks.

    Attributes:
        - K: Number of Gaussian kernels.
        - means, stds: Embeddings for the means and standard deviations of the Gaussian kernels.
        - mul, bias: Embeddings for scaling and bias parameters.
    """
    def __init__(self, K=128, edge_types=1024):
        """
        Initializes the GaussianLayer module.

        :param K: Number of Gaussian kernels.
        :param edge_types: Number of different edge types to consider.

        :return: An instance of the configured Gaussian kernel and edge types.
        """
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        """
        Forward pass of the GaussianLayer.

        :param x: Input tensor representing distances or other features.
        :param edge_type: Tensor indicating types of edges in the graph.

        :return: Tensor transformed by the Gaussian layer.
        """
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
    
class NumericalEmbed(nn.Module):
    """
    Numerical embedding module, typically used for embedding edge features in graph neural networks.

    Attributes:
        - K: Output dimension for embeddings.
        - mul, bias, w_edge: Embeddings for transformation parameters.
        - proj: Projection layer to transform inputs.
        - ln: Layer normalization.
    """
    def __init__(self, K=128, edge_types=1024, activation_fn='gelu'):
        """
        Initializes the NonLinearHead.

        :param input_dim: The input dimension of the first layer.
        :param out_dim: The output dimension of the second layer.
        :param activation_fn: The activation function to use.
        :param hidden: The dimension of the hidden layer; defaults to input_dim if not specified.
        """
        super().__init__()
        self.K = K 
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        self.w_edge = nn.Embedding(edge_types, K)

        self.proj = NonLinearHead(1, K, activation_fn, hidden=2*K)
        self.ln = LayerNorm(K)

        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        nn.init.kaiming_normal_(self.w_edge.weight)


    def forward(self, x, edge_type):    # edge_type, atoms
        """
        Forward pass of the NonLinearHead.

        :param x: Input tensor to the classification head.

        :return: The output tensor after passing through the layers.
        """
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        w_edge = self.w_edge(edge_type).type_as(x)
        edge_emb = w_edge * torch.sigmoid(mul * x.unsqueeze(-1) + bias)
        
        edge_proj = x.unsqueeze(-1).type_as(self.mul.weight)
        edge_proj = self.proj(edge_proj)
        edge_proj = self.ln(edge_proj)

        h = edge_proj + edge_emb
        h = h.type_as(self.mul.weight)
        return h

def molecule_architecture():
    args = argparse.ArgumentParser()
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.2)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", "gaussian")
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    return args

def fds_config(): 
    # feature_dim=512, bucket_num=100, bucket_start=0,
                #start_update=0, start_smooth=1, kernel='gaussian', ks=5, sigma=1, momentum=0.9
    args = argparse.ArgumentParser()
    args.feature_dim = getattr(args, "feature_dim", 512)
    args.bucket_num = getattr(args, "bucket_num", 20)
    args.bucket_start = getattr(args, "bucket_start", 0)
    args.start_update = getattr(args, "start_update", 0)
    args.start_smooth = getattr(args, "start_smooth", 1)
    args.kernel = getattr(args, "kernel", 'gaussian')
    args.ks = getattr(args, "ks", 5)
    args.sigma = getattr(args, "sigma", 1)
    args.momentum = getattr(args, "momentum", 0.9)
    args.col_data = getattr(args, "col_data", "expt")
    args.raw_data = getattr(args, "raw_data","")
    return args  

def crossmodal_config():
    # Initialize the parser
    args = argparse.ArgumentParser()
    args.attention_probs_dropout_prob = getattr(args, "attention_probs_dropout_prob", 0.2)
    args.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)
    args.hidden_act = getattr(args, "hidden_act", "gelu")
    args.hidden_dropout_prob = getattr(args, "hidden_dropout_prob", 0.3)
    args.hidden_size = getattr(args, "hidden_size", 512)
    args.initializer_range = getattr(args, "initializer_range", 0.02)
    args.intermediate_size = getattr(args, "intermediate_size", 2048)
    args.layer_norm_eps = getattr(args, "layer_norm_eps", 1e-12)
    args.max_position_embeddings = getattr(args, "max_position_embeddings", 512)
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)
    args.num_hidden_layers = getattr(args, "num_hidden_layers", 12)
    args.position_embedding_type = getattr(args, "position_embedding_type", "absolute")
    return args

class CrossAttentionModel(nn.Module):
    def __init__(self, cross_cfg, num_layers=1):
        super(CrossAttentionModel, self).__init__()
        self.text_attention = BertCrossEncoder(cross_cfg, num_layers)
        self.graph_attention = BertCrossEncoder(cross_cfg, num_layers)
        self.dropout = nn.Dropout(cross_cfg.hidden_dropout_prob)

    def forward(self, text_embeddings, graph_embeddings, text_mask, graph_mask):
        # Apply layer normalization before attention
        # text_embeddings = self.text_layernorm(text_embeddings)
        # graph_embeddings = self.graph_layernorm(graph_embeddings)
        text_embeddings = self.dropout(text_embeddings)
        graph_embeddings = self.dropout(graph_embeddings)

        # Cross attention: text attends to graph and graph attends to text
        extended_txt_mask = text_mask.unsqueeze(1).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        # Cross-attention
        cross_txt_encoder = self.graph_attention(graph_embeddings, text_embeddings, extended_txt_mask)
        graph_to_text = cross_txt_encoder[-1]

        extended_img_mask = graph_mask.unsqueeze(1).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.text_attention(text_embeddings, graph_embeddings, extended_img_mask)
        text_to_graph = cross_encoder[-1]
        return text_to_graph, graph_to_text

class MM_Model(nn.Module):
    def __init__(self, output_dim=2, **params):
        """
        Initializes the UniMolModel with specified parameters and data type.

        :param output_dim: (int) The number of output dimensions (classes).
        :param data_type: (str) The type of data (e.g., 'molecule', 'protein').
        :param params: Additional parameters for model configuration.
        """
        super().__init__()
        data_type='molecule'
        self.cross_cfg= crossmodal_config()
        self.fds_cfg= fds_config()
        self.args = molecule_architecture()
        self.output_dim = output_dim
        self.data_type = data_type
        self.remove_hs = params.get('remove_hs', False)
        self.use_fds = params.get('fds', False)
        self.using_scale = params.get('use_scaler', True)
        self.fds_num = params.get('fds_num', 30)
        self.fds_raw_path = params.get('fds_raw_path', '')
        self.fds_col_data = params.get('fds_col_data', '')
        self.ct_w = params.get('ct_w', 0.2)
        self.task = params.get('task', False)
        self.chemberta_dir = params.get('chemberta_dir', '')
        self.unimol_dir = params.get('unimol_dir', '')
        
        self.dictionary = Dictionary.load(os.path.join(os.path.dirname(self.unimol_dir), 'mol.dict.txt'))

        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = self.dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(self.dictionary), self.args.encoder_embed_dim, self.padding_idx
        )
        self.encoder = BACKBONE[self.args.backbone](
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.delta_pair_repr_norm_loss < 0,
        )
        K = 128
        n_edge_type = len(self.dictionary) * len(self.dictionary)
        self.gbf_proj = NonLinearHead(
            K, self.args.encoder_attention_heads, self.args.activation_fn
        )
        if self.args.kernel == 'gaussian':
            self.gbf = GaussianLayer(K, n_edge_type)
        else:
            self.gbf = NumericalEmbed(K, n_edge_type)
        
        self.classification_head = ClassificationHead(
            input_dim=self.cross_cfg.hidden_size,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=self.output_dim,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        self.apply(init_bert_params)
        self.load_pretrained_weights(path=self.unimol_dir)

        self.bert= AutoModel.from_pretrained(self.chemberta_dir)
        self.tokenizer= AutoTokenizer.from_pretrained(self.chemberta_dir)

        self.cross_modal_module= CrossAttentionModel(self.cross_cfg, num_layers=1)

        # Select the appropriate ConR class
        if self.task == 'classification':
            # from .conr_class import ConR_single as ConR\
            from .contrastive import CT_Single as CT
        elif self.task == 'multilabel_classification':
            # from .conr_class import ConR_Multi as ConR
            from .contrastive import CT_Multi as CT
        elif self.task == 'regression':
            # from .conr import ConR
            from .contrastive import CT_Regress as CT

        self.CT = CT

        self.infonce= InfoNCE(self.cross_cfg.hidden_size, self.cross_cfg.hidden_size)
        if self.use_fds and self.task == 'regression':
            self.FDS = FDS(raw_data=self.fds_raw_path, using_scale=self.using_scale, col_data=self.fds_col_data, feature_dim=self.fds_cfg.feature_dim, bucket_num=self.fds_num, bucket_start=self.fds_cfg.bucket_start,
                        start_update=self.fds_cfg.start_update, start_smooth=self.fds_cfg.start_smooth, kernel=self.fds_cfg.kernel, ks=self.fds_cfg.ks, 
                        sigma=self.fds_cfg.sigma, momentum=self.fds_cfg.momentum)

    def load_pretrained_weights(self, path):
        """
        Loads pretrained weights into the model.

        :param path: (str) Path to the pretrained weight file.
        """
        if path is not None:
            if self.data_type == 'mof':
                logger.info("Loading pretrained weights from {}".format(path))
                state_dict = torch.load(path, map_location=lambda storage, loc: storage)
                model_dict = {k.replace('unimat.',''):v for k, v in state_dict['model'].items()}
                self.load_state_dict(model_dict, strict=True)
            else:
                logger.info("Loading pretrained weights from {}".format(path))
                state_dict = torch.load(path, map_location=lambda storage, loc: storage)
                self.load_state_dict(state_dict['model'], strict=False)

    @classmethod
    def build_model(cls, args):
        """
        Class method to build a new instance of the UniMolModel.

        :param args: Arguments for model configuration.
        :return: An instance of UniMolModel.
        """
        return cls(args)
           
    def forward(
        self,
        src_tokens,
        src_distance,
        src_edge_type,
        input_ids,
        attention_mask,
        weights=None,
        return_infonce_loss=False,
        return_ct_loss=False,
        return_feature=False,
        net_target=None,
        use_weight=None,
        epoch=0,
        **kwargs
    ):

        
        # Prepare masks
        padding_mask = src_tokens.eq(self.padding_idx)
        img_mask = ~padding_mask
        attention_mask = attention_mask.bool().to(src_tokens.device)
        if not padding_mask.any():
            padding_mask = None

        # Embeddings
        x = self.embed_tokens(src_tokens)
        graph_attn_bias = self.gbf(src_distance, src_edge_type)
        graph_attn_bias = self.gbf_proj(graph_attn_bias)
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, graph_attn_bias.size(-2), graph_attn_bias.size(-1))

        # Encoder
        encoder_rep, _, _, _, _ = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)

        # BERT
        out_bert = self.bert(input_ids, attention_mask, return_dict=True)[0]

        # Contrastive loss
        # if return_infonce_loss:
        if return_infonce_loss:
            ct_loss = self.infonce(encoder_rep, out_bert)

        # Cross-modal fusion: text_embeddings, graph_embeddings, text_mask, graph_mask
        # output: text_to_graph, graph_to_text
        cross_txt_output_layer, cross_output_layer = self.cross_modal_module(encoder_rep, out_bert, img_mask, attention_mask)
        cross_txt_output_layer[~img_mask] = 0.0
        cross_output_layer[~attention_mask] = 0.0
        # Gated fusion
        final_output = torch.cat((cross_txt_output_layer, cross_output_layer), dim=1)
        classification_feats_pooled = final_output.sum(dim=1) / ((img_mask).sum(dim=1).view(-1, 1) + (attention_mask).sum(dim=1).view(-1, 1))

        # Smooth features
        smoothed_features = classification_feats_pooled
        if self.training and epoch >= self.fds_cfg.start_smooth and self.use_fds and self.task == 'regression':
            smoothed_features = self.FDS.smooth(smoothed_features, net_target, epoch)

        logits = self.classification_head(smoothed_features)

        if not return_feature:
            if return_infonce_loss:
                if return_ct_loss and net_target is not None: 
                    if use_weight:
                        rnc_loss= self.CT(classification_feats_pooled, net_target, logits, weights= weights, w= self.ct_w)
                    else:
                        rnc_loss= self.CT(classification_feats_pooled, net_target, logits, w= self.ct_w)
                    return logits, ct_loss, rnc_loss
                return logits, ct_loss
            else:
                if return_ct_loss and net_target is not None: 
                    if use_weight:
                        rnc_loss= self.CT(classification_feats_pooled, net_target, logits, weights= weights, w= self.ct_w)
                    else:
                        rnc_loss= self.CT(classification_feats_pooled, net_target, logits, w= self.ct_w)
                    return logits, rnc_loss
            return logits
        else:
            if return_infonce_loss:
                if return_ct_loss and net_target is not None: 
                    if use_weight:
                        rnc_loss= self.CT(classification_feats_pooled, net_target, logits, weights= weights, w= self.ct_w)
                    else:
                        rnc_loss= self.CT(classification_feats_pooled, net_target, logits, w= self.ct_w)
                    return logits, classification_feats_pooled, ct_loss, rnc_loss
                return logits, classification_feats_pooled, ct_loss
            else:
                if return_ct_loss and net_target is not None: 
                    if use_weight:
                        rnc_loss= self.CT(classification_feats_pooled, net_target, logits, weights= weights, w= self.ct_w)
                    else:
                        rnc_loss= self.CT(classification_feats_pooled, net_target, logits, w= self.ct_w)
                    return logits, classification_feats_pooled, rnc_loss
            return logits, classification_feats_pooled   
    def batch_collate_fn_mof(self, samples):
        """
        Custom collate function for batch processing MOF data.

        :param samples: A list of sample data.

        :return: A batch dictionary with padded and processed features.
        """
        dd = {}
        for k in samples[0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'gas_id':
                v = torch.tensor([s[k] for s in samples]).long()
            elif k in ['gas_attr', 'temperature', 'pressure']:
                v = torch.tensor([s[k] for s in samples]).float()
            else:
                continue
            dd[k] = v
        return dd
    def batch_collate_fn(self, samples):
        """
        Custom collate function for batch processing non-MOF data.

        :param samples: A list of sample data.

        :return: A tuple containing a batch dictionary and labels.
        """
        batch = {}
        # print(samples[0][0].keys())
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k=='weights':
                v = torch.tensor([s[0][k] for s in samples])
                
            if k != 'smile':
                batch[k] = v

        if 'smile' in samples[0][0].keys():
            batch_text = self.tokenizer([i[0]['smile'] for i in samples], padding=True, truncation=True, return_tensors="pt")
            batch_text_inputids = batch_text['input_ids']
            batch_text_attention = batch_text['attention_mask']
            batch['input_ids'] = batch_text_inputids
            batch['attention_mask'] = batch_text_attention

        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        # print(batch.keys())
        return batch, label
    