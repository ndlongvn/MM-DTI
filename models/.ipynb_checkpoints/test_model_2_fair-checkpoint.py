
from __future__ import absolute_import, division, print_function
from ast import Not
from .test_module import BertSelfEncoder, BertCrossEncoder, create_mask, ActivateFun, BertConfig
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


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
import argparse
import pathlib
import os
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass, field
from typing import Optional
from .fds import FDS
from numpy.lib.arraysetops import isin
from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models.doublemodel import TrEncoder
from fairseq import data 
from ..utils import logger
from ..config import MODEL_CONFIG
import pickle
from .rnc_loss import ConR

data_dictionary= data.Dictionary.load(os.path.join("/workspace1/longnd38/unimol/Uni-Mol/DVMP_1/data/preprocess/covid19", "input0", "dict.txt"))
@dataclass
class OneModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(default='gelu', )
    dropout: float = field(default=0.1)
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)
    relu_dropout: float = field(default=0.0)
    encoder_embed_path: Optional[str] = field(default=None)
    encoder_embed_dim: int = field(default=768)
    encoder_ffn_embed_dim: int = field(default=3072)
    encoder_layers: int = field(default=12)
    encoder_attention_heads: int = field(default=12)
    encoder_normalize_before: bool = field(default=False)
    encoder_learned_pos: bool = field(default=True)
    layernorm_embedding: bool = field(default=True)
    no_scale_embedding: bool = field(default=True)
    max_positions: int = field(default=512)

    gnn_number_layer: int = field(default=12)
    gnn_dropout: float = field(default=0.1)
    conv_encode_edge: bool = field(default=True)
    gnn_embed_dim: int = field(default=384)
    gnn_aggr: str = field(default='maxminmean')
    gnn_norm: str = field(default='batch')
    gnn_activation_fn: str = field(default='relu')

    datatype: str = field(default='g')
    classification_head_name: str = field(default='')
    load_checkpoint_heads: bool = field(default=False)
    gradmultiply: bool = field(default=False)

    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={"help": "scalar quantization noise and scalar quantization at training time"},
    )
    # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
    spectral_norm_classification_head: bool = field(default=False)

    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(default=0.0,
                                     metadata={"help": "LayerDrop probability for decoder"})
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={"help": "which layers to *keep* when pruning as a comma-separated list"},
    )
    max_source_positions: int = field(default=512) #II("model.max_positions")
    no_token_positional_embeddings: bool = field(default=False)
    pooler_activation_fn: str = field(default='tanh')
    pooler_dropout: float = field(default=0.0)
    untie_weights_roberta: bool = field(default=False)
    adaptive_input: bool = field(default=False)

     

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

class TestModel_2_Fair(nn.Module):
    """
    UniMolModel is a specialized model for molecular, protein, crystal, or MOF (Metal-Organic Frameworks) data. 
    It dynamically configures its architecture based on the type of data it is intended to work with. The model
    supports multiple data types and incorporates various architecture configurations and pretrained weights.

    Attributes:
        - output_dim: The dimension of the output layer.
        - data_type: The type of data the model is designed to handle.
        - remove_hs: Flag to indicate whether hydrogen atoms are removed in molecular data.
        - pretrain_path: Path to the pretrained model weights.
        - dictionary: The dictionary object used for tokenization and encoding.
        - mask_idx: Index of the mask token in the dictionary.
        - padding_idx: Index of the padding token in the dictionary.
        - embed_tokens: Embedding layer for token embeddings.
        - encoder: Transformer encoder backbone of the model.
        - gbf_proj, gbf: Layers for Gaussian basis functions or numerical embeddings.
        - classification_head: The final classification head of the model.
    """
    def __init__(self, output_dim=2, config=None,**params):
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

        self.bert=  TrEncoder(OneModelConfig, data_dictionary)
        self.bert= load_model_2("/workspace1/longnd38/DVMP/checkpoints/checkpoint_best.pt", self.bert)
        self.bert_pro= nn.Linear(768, 512)
        self.dropout = nn.Dropout(self.cross_cfg.hidden_dropout_prob)
        self.txt2img_attention = BertCrossEncoder(self.cross_cfg, 1)
        self.img2txt_attention = BertCrossEncoder(self.cross_cfg, 1)

        self.fuse_type = 'mean'
        self.infonce= InfoNCE(self.cross_cfg.hidden_size, self.cross_cfg.hidden_size)
        self.infonce1= InfoNCE(self.cross_cfg.hidden_size, self.cross_cfg.hidden_size)
        self.FDS = FDS(raw_data=self.fds_cfg.raw_data, feature_dim=self.fds_cfg.feature_dim, bucket_num=self.fds_cfg.bucket_num, bucket_start=self.fds_cfg.bucket_start,
                        start_update=self.fds_cfg.start_update, start_smooth=self.fds_cfg.start_smooth, kernel=self.fds_cfg.kernel, ks=self.fds_cfg.ks, 
                        sigma=self.fds_cfg.sigma, momentum=self.fds_cfg.momentum)


        
        # self.self_attention_v2 = BertSelfEncoder(config)
 

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
        attention_mask= input_ids.ne(data_dictionary.pad())
        out_bert = self.bert(input_ids, return_all_hiddens=True)[0]
        out_bert= self.bert_pro(out_bert)
        # print("out_bert", out_bert.shape)
        # print("out_bert: ",out_bert.shape)
        # contrastive loss
        if return_ct_loss:
            ct_loss= self.infonce(all_repr, out_bert)+ self.infonce1(out_bert, all_repr)

        #  B, L, 384
        """for bert"""
        text_output = self.dropout(out_bert)
        extended_txt_mask = attention_mask
        # main_addon_sequence_encoder = text_output
        main_addon_sequence_output = text_output
        # print("main_addon_sequence_output: ",main_addon_sequence_output.shape)

        audio_output= all_repr #.clone()

        converted_vis_embed_map_v2 = audio_output #self.vismap2text(audio_output)

        # print("cross_txt_output_layer: ",cross_txt_output_layer.shape)
        """for graph"""
        # from 512 to 384
        converted_vis_embed_map = audio_output #self.vismap2text_v2(audio_output)

        img_mask = (1-padding_mask.to(dtype=next(self.parameters()).dtype)) # .squeeze(1)
        # calculate extended_img_mask required for cross-attention
        extended_img_mask = img_mask
        
            ### query: graph, key, value: text

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_img_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]
            ### query: text, key, value: graph
        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_txt_mask)
        cross_output_layer = cross_encoder[-1]

        # final_output = torch.cat((cross_txt_output_layer, cross_output_layer), dim=1)                                    

        if self.fuse_type=='mean':
            cross_txt_output_layer[padding_mask]=0.0
            cross_output_layer[~attention_mask]=0.0
            final_output = torch.cat((cross_txt_output_layer, cross_output_layer), dim=1)
            # classification_feats_pooled = torch.mean(final_output, dim=1)
            classification_feats_pooled = final_output.sum(dim=1)/((~padding_mask).sum(dim=1).view(-1, 1)+ (attention_mask).sum(dim=1).view(-1, 1))

            
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
        # print(samples)
        # with open("test.pkl", "wb") as output_file:
        #        pickle.dump(samples, output_file)
            
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            if k != 'smile':
                batch[k] = v

        batch['input_ids']= pad_1d_tokens([torch.tensor(s[0]['net_input']).long() for s in samples], pad_idx=data_dictionary.pad())

        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        # print(batch.keys())
        return batch, label
    
def load_model_2(load_path, model, allow_size_mismatch = True): # , optimizer = None
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile( load_path ), "Error when loading the model, provided path not found: {}".format( load_path )
    checkpoint = torch.load(load_path)
    try:
        loaded_state_dict = checkpoint['model']
    except:
        loaded_state_dict = checkpoint['model_state_dict']
    loaded_state_dict = { k: v for k,v in loaded_state_dict.items() if "encoder0." in k}
    loaded_state_dict = { k[9:]: v for k,v in loaded_state_dict.items()} # if 'trunk' not in k 
    if allow_size_mismatch:
        loaded_sizes = { k: v.shape for k,v in loaded_state_dict.items() }
        model_state_dict = model.state_dict()
        model_sizes = { k: v.shape for k,v in model_state_dict.items() }
        mismatched_params = []
        for k in loaded_sizes:
            if k in model_sizes:
                if loaded_sizes[k] != model_sizes[k]:
                    mismatched_params.append(k)
            else:
                mismatched_params.append(k)
        for k in mismatched_params:
            del loaded_state_dict[k]
        print('mismatched_params: ', mismatched_params)
    # -- copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict, strict = not allow_size_mismatch)
    print("Load model {}".format(load_path))
    return model #, wer, cer