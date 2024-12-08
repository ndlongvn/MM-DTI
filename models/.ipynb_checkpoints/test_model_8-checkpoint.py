
from __future__ import absolute_import, division, print_function
from ast import Not
from .test_module import BertSelfOutput, BertIntermediate, BertOutput
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore.utils import get_activation_fn
from unicore.data import Dictionary
from unicore.models import BaseUnicoreModel
from unicore.modules import LayerNorm, init_bert_params
from .transformers import TransformerEncoderWithPair
from ..utils import pad_1d_tokens, pad_2d, pad_coords
from .infonce import InfoNCE
from .rnc_loss import ConR
from .fds import FDS
import argparse
import pathlib
import os
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, AutoConfig
# from transformers import AutoModel, AutoConfig
import warnings
warnings.filterwarnings("ignore")
tokenizer= AutoTokenizer.from_pretrained("/workspace1/longnd38/unimol/MLM/MLP_TrainedModels")

from ..utils import logger
from ..config import MODEL_CONFIG

BACKBONE = {
    'transformer': TransformerEncoderWithPair,
}

WEIGHT_DIR = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'weights')

class AgentAttention(nn.Module):
    def __init__(self, config, attn_drop=0., proj_drop=0.,
                 agent_num=16, **kwargs):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))


        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(config.hidden_size, self.dim)
        self.key = nn.Linear(config.hidden_size, self.dim)
        self.value = nn.Linear(config.hidden_size, self.dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num

        self.pool = nn.AdaptiveAvgPool1d(output_size=(agent_num))
        self.an_bias = nn.Parameter(torch.zeros(self.num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(self.num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, self.num_heads, agent_num, self.agent_num, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, self.num_heads, agent_num, 1, self.agent_num))
        self.ha_bias = nn.Parameter(torch.zeros(1, self.num_heads, self.agent_num, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.agent_num, agent_num))
        nn.init.trunc_normal_(self.an_bias, std=.02)
        nn.init.trunc_normal_(self.na_bias, std=.02)
        nn.init.trunc_normal_(self.ah_bias, std=.02)
        nn.init.trunc_normal_(self.aw_bias, std=.02)
        nn.init.trunc_normal_(self.ha_bias, std=.02)
        nn.init.trunc_normal_(self.wa_bias, std=.02)
        

    def transpose_for_scores(self, x):
        # b, l, c
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def transpose_for_bias(self, x):
        # x: b, c, l
        new_x_shape = x.size()[:-1] + (self.num_heads, self.agent_num)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)


    def forward(self, s1_hidden_states, s2_hidden_states, s1_attention_mask= None, s2_attention_mask= None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = s1_hidden_states.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        q = self.query(s1_hidden_states) # b, n, c
        k = self.key(s2_hidden_states)
        v = self.value(s2_hidden_states)

        agent_tokens = self.pool(q.permute(0, 2, 1))
        agent_tokens= agent_tokens.reshape(b, c, -1).permute(0, 2, 1)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.agent_num, self.agent_num), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        position_bias= torch.mean(position_bias, dim= -1)
        
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        if s2_attention_mask!=None:
            agent_attn= (agent_tokens * self.scale) @ k.transpose(-2, -1)+ s2_attention_mask#+ torch.rand(b, 1, 512)
        else:
            agent_attn = (agent_tokens * self.scale) @ k.transpose(-2, -1)

        agent_attn = (agent_attn.transpose(-2, -1) + position_bias.unsqueeze(2)).transpose(-2, -1)
        agent_attn = self.softmax(agent_attn)

        agent_attn = self.attn_drop(agent_attn)

        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.agent_num, self.agent_num), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        agent_bias= torch.mean(agent_bias, dim= -2)
        if s1_attention_mask!=None:
            q_attn = (q * self.scale) @ agent_tokens.transpose(-2, -1)
            q_attn = q_attn+ s1_attention_mask.transpose(-2, -1) # + agent_bias
        else:
            q_attn = (q * self.scale) @ agent_tokens.transpose(-2, -1)

        q_attn= q_attn + agent_bias.unsqueeze(2)
        q_attn = self.softmax(q_attn)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v 
        x = x.transpose(1, 2).reshape(b, n, c)
  
        return x
    
class BertAgentAttention(nn.Module):
    def __init__(self, config):
        super(BertAgentAttention, self).__init__()
        self.self = AgentAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s1_attention_mask=None, s2_attention_mask=None):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s1_attention_mask, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output
    
class BertAgentAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertAgentAttentionLayer, self).__init__()
        self.attention = BertAgentAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s1_attention_mask=None, s2_attention_mask=None):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s1_attention_mask,  s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertAgentEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertAgentEncoder, self).__init__()
        layer = BertAgentAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s1_attention_mask=None, s2_attention_mask=None, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s1_attention_mask, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers

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
    args.bucket_num = getattr(args, "bucket_num", 200)
    args.bucket_start = getattr(args, "bucket_start", 3)
    args.start_update = getattr(args, "start_update", 0)
    args.start_smooth = getattr(args, "start_smooth", 1)
    args.kernel = getattr(args, "kernel", 'gaussian')
    args.ks = getattr(args, "ks", 5)
    args.sigma = getattr(args, "sigma", 1)
    args.momentum = getattr(args, "momentum", 0.9)
    args.raw_data = getattr(args, "raw_data", "/workspace1/longnd38/unimol/Uni-Mol/unimol_tools/data/train_split.csv")
    
    return args  

def crossmodal_config():
    # Initialize the parser
    args = argparse.ArgumentParser()
    args.attention_probs_dropout_prob = getattr(args, "attention_probs_dropout_prob", 0.1)
    args.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)
    args.hidden_act = getattr(args, "hidden_act", "gelu")
    args.hidden_dropout_prob = getattr(args, "hidden_dropout_prob", 0.1)
    args.hidden_size = getattr(args, "hidden_size", 512)
    args.initializer_range = getattr(args, "initializer_range", 0.02)
    args.intermediate_size = getattr(args, "intermediate_size", 2048)
    args.layer_norm_eps = getattr(args, "layer_norm_eps", 1e-12)
    args.max_position_embeddings = getattr(args, "max_position_embeddings", 512)
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)
    args.num_hidden_layers = getattr(args, "num_hidden_layers", 12)
    args.position_embedding_type = getattr(args, "position_embedding_type", "absolute")
    return args
class TestModel_8(nn.Module):

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
        if data_type == 'molecule':
            name = "no_h" if self.remove_hs else "all_h" 
            name = data_type + '_' + name
            self.pretrain_path = os.path.join(WEIGHT_DIR, MODEL_CONFIG['weight'][name])
            self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, MODEL_CONFIG['dict'][name]))
        else:
            self.pretrain_path = os.path.join(WEIGHT_DIR, MODEL_CONFIG['weight'][data_type])
            self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, MODEL_CONFIG['dict'][data_type]))
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
        self.load_pretrained_weights(path=self.pretrain_path)

        self.bert= AutoModel.from_pretrained("/workspace1/longnd38/unimol/MLM/MLP_TrainedModels")

        self.dropout = nn.Dropout(self.cross_cfg.hidden_dropout_prob)
        self.txt2img_attention = BertAgentEncoder(self.cross_cfg, 1)
        self.img2txt_attention = BertAgentEncoder(self.cross_cfg, 1)

        self.fuse_type = 'mean'
        self.infonce= InfoNCE(self.cross_cfg.hidden_size, self.cross_cfg.hidden_size)
        self.infonce1= InfoNCE(self.cross_cfg.hidden_size, self.cross_cfg.hidden_size)
        self.FDS = FDS(raw_data=self.fds_cfg.raw_data, feature_dim=self.fds_cfg.feature_dim, bucket_num=self.fds_cfg.bucket_num, bucket_start=self.fds_cfg.bucket_start,
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
        weights,
        return_ct_loss=False,
        return_rnc_loss=False,
        return_feature=False,
        net_target= None,
        use_weight= None,
        epoch=0,
        **kwargs
    ):
        """
        Defines the forward pass of the model.

        :param src_tokens: Tokenized input data.
        :param src_distance: Additional molecular features.
        :param src_coord: Additional molecular features.
        :param src_edge_type: Additional molecular features.
        :param gas_id: Optional environmental features for MOFs.
        :param gas_attr: Optional environmental features for MOFs.
        :param pressure: Optional environmental features for MOFs.
        :param temperature: Optional environmental features for MOFs.
        :param return_repr: Flags to return intermediate representations.
        :param return_atomic_reprs: Flags to return intermediate representations.

        :return: Output logits or requested intermediate representations.
        """
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            _,
            _,
            _,
            _,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_rep[:, 0, :]  # B, L, 512# CLS token repr
        all_repr = encoder_rep[:, :, :]
        # print("all_repr: ",all_repr.shape)
        """change"""
        
        # add bert
        out_bert = self.bert(input_ids, attention_mask, return_dict=True)[0] 
        # print("out_bert: ",out_bert.shape)
        # contrastive loss
        if return_ct_loss:
            ct_loss= self.infonce(all_repr, out_bert)+ self.infonce1(out_bert, all_repr)

        #  B, L, 384
        """for bert"""
        text_output = self.dropout(out_bert)
        extended_txt_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        # main_addon_sequence_encoder = text_output
        main_addon_sequence_output = text_output
        # print("main_addon_sequence_output: ",main_addon_sequence_output.shape)

        audio_output= all_repr #.clone()

        converted_vis_embed_map_v2 = audio_output #self.vismap2text(audio_output)

        # print("cross_txt_output_layer: ",cross_txt_output_layer.shape)
        """for graph"""
        # from 512 to 384
        converted_vis_embed_map = audio_output #self.vismap2text_v2(audio_output)

        img_mask = (1-padding_mask.clone().to(dtype=next(self.parameters()).dtype)) # .squeeze(1)
        # calculate extended_img_mask required for cross-attention
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0
        
            ### query: graph, key, value: text

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output)
        cross_txt_output_layer = cross_txt_encoder[-1]
            ### query: text, key, value: graph
        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map)
        cross_output_layer = cross_encoder[-1]

        # final_output = torch.cat((cross_txt_output_layer, cross_output_layer), dim=1)                                    

        # if self.fuse_type=='mean':
        #     cross_txt_output_layer[padding_mask]=0.0
        #     cross_output_layer[~attention_mask]=0.0
        #     final_output = torch.cat((cross_txt_output_layer, cross_output_layer), dim=1)
        #     # classification_feats_pooled = torch.mean(final_output, dim=1)
        #     classification_feats_pooled = final_output.sum(dim=1)/((~padding_mask).sum(dim=1).view(-1, 1)+ (attention_mask).sum(dim=1).view(-1, 1))
        final_output = torch.cat((cross_txt_output_layer, cross_output_layer), dim=1)                                    

        if self.fuse_type=='mean':
            classification_feats_pooled = torch.mean(final_output, dim=1)
        elif self.fuse_type=='max':
            classification_feats_pooled,_ = torch.max(final_output,dim = 1)
        elif self.fuse_type=='cls':
            classification_feats_pooled = final_output[:,0,:]
            
        smoothed_features = classification_feats_pooled    
        if self.training and epoch >= self.fds_cfg.start_smooth:
            smoothed_features = self.FDS.smooth(smoothed_features, net_target, epoch)
        # preds = self.regressor(smoothed_features)
        logits = self.classification_head(smoothed_features)

        if not return_feature:
            if return_ct_loss:
                # print("return_ct_loss")
                if return_rnc_loss and net_target is not None: 
                    if use_weight:
                        rnc_loss= ConR(classification_feats_pooled, net_target, logits, weights= weights)
                    else:
                        rnc_loss= ConR(classification_feats_pooled, net_target, logits)
                    return logits, ct_loss, rnc_loss
                return logits, ct_loss
            else:
                if return_rnc_loss and net_target is not None: 
                    if use_weight:
                        # print(weights)
                        rnc_loss= ConR(classification_feats_pooled, net_target, logits, weights= weights)
                    else:
                        rnc_loss= ConR(classification_feats_pooled, net_target, logits)
                    return logits, rnc_loss
            return logits
        else:
            if return_ct_loss:
                # print("return_ct_loss")
                if return_rnc_loss and net_target is not None: 
                    if use_weight:
                        rnc_loss= ConR(classification_feats_pooled, net_target, logits, weights= weights)
                    else:
                        rnc_loss= ConR(classification_feats_pooled, net_target, logits)
                    return logits, classification_feats_pooled, ct_loss, rnc_loss
                return logits, classification_feats_pooled, ct_loss
            else:
                if return_rnc_loss and net_target is not None: 
                    if use_weight:
                        # print(weights)
                        rnc_loss= ConR(classification_feats_pooled, net_target, logits, weights= weights)
                    else:
                        rnc_loss= ConR(classification_feats_pooled, net_target, logits)
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
            batch_text = tokenizer([i[0]['smile'] for i in samples], padding=True, return_tensors="pt")
            batch_text_inputids = batch_text['input_ids']
            batch_text_attention = batch_text['attention_mask']
            batch['input_ids'] = batch_text_inputids
            batch['attention_mask'] = batch_text_attention
            # print(samples[0][0]['smile'])

        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        # print(batch.keys())
        return batch, label
    


class Chemberta(nn.Module):
    def __init__(self, model_name_or_path= '/workspace1/longnd38/unimol/chemberta', num_labels=18, **params):
        super(Chemberta, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.linear = ClassificationHead(
                input_dim=self.config.hidden_size,
                inner_dim=self.config.hidden_size,
                num_classes=num_labels,
                activation_fn='tanh',
                pooler_dropout=0.2,
            )
        self.tokenizer= AutoTokenizer.from_pretrained(model_name_or_path)
        # nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs[0][:, 0, :]  # Access CLS token representation
        logits = self.linear(cls_token)
        return logits
    
    def batch_collate_fn(self, samples):
        batch= {}
        if 'smile' in samples[0][0].keys():
            batch_text = self.tokenizer([i[0]['smile'] for i in samples], padding=True, return_tensors="pt")
            batch_text_inputids = batch_text['input_ids']
            batch_text_attention = batch_text['attention_mask']
            batch['input_ids'] = batch_text_inputids
            batch['attention_mask'] = batch_text_attention
            # print(samples[0][0]['smile'])

        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        # print(batch.keys())
        return batch, label
    
class ChembertaCustom(nn.Module):
    def __init__(self, model_name_or_path= '/workspace1/longnd38/unimol/MLM/MLP_TrainedModels', num_labels=18, **params):
        super(ChembertaCustom, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.linear = ClassificationHead(
                input_dim=self.config.hidden_size,
                inner_dim=self.config.hidden_size,
                num_classes=num_labels,
                activation_fn='tanh',
                pooler_dropout=0.2,
            )
        self.tokenizer= AutoTokenizer.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs[0][:, 0, :]  # Access CLS token representation
        logits = self.linear(cls_token)
        return logits
    
    def batch_collate_fn(self, samples):
        batch= {}
        if 'smile' in samples[0][0].keys():
            batch_text = self.tokenizer([i[0]['smile'] for i in samples], padding=True, return_tensors="pt")
            batch_text_inputids = batch_text['input_ids']
            batch_text_attention = batch_text['attention_mask']
            batch['input_ids'] = batch_text_inputids
            batch['attention_mask'] = batch_text_attention
            # print(samples[0][0]['smile'])

        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        # print(batch.keys())
        return batch, label