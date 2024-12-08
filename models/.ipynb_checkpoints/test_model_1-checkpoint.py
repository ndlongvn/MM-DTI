
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
tokenizer= AutoTokenizer.from_pretrained("/workspace1/longnd38/unimol/chemberta")

from ..utils import logger
from ..config import MODEL_CONFIG

BACKBONE = {
    'transformer': TransformerEncoderWithPair,
}

WEIGHT_DIR = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'weights')

class UniMolModel(BaseUnicoreModel):
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
    def __init__(self, output_dim=2, data_type='molecule', **params):
        """
        Initializes the UniMolModel with specified parameters and data type.

        :param output_dim: (int) The number of output dimensions (classes).
        :param data_type: (str) The type of data (e.g., 'molecule', 'protein').
        :param params: Additional parameters for model configuration.
        """
        super().__init__()
        if data_type == 'molecule':
            self.args = molecule_architecture()
        elif data_type == 'oled':
            self.args = oled_architecture()
        elif data_type == 'protein':
            self.args = protein_architecture()
        elif data_type == 'crystal':
            self.args = crystal_architecture()
        elif data_type == 'mof':
            self.args = mof_architecture()
        else:
            raise ValueError('Current not support data type: {}'.format(data_type))
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
        
        if data_type == 'mof':
            self.min_max_key = {
                'pressure': [-4.0, 6.0],      # transoformed pressure in log10(P)
                'temperature': [100, 400.0],  
             }
            self.gas_embed = GasModel(self.args.gas_attr_input_dim, self.args.hidden_dim)
            self.env_embed = EnvModel(self.args.hidden_dim, self.args.bins, self.min_max_key)
            self.classifier = ClassificationHead(self.args.encoder_embed_dim+self.args.hidden_dim*5, 
                                    self.args.hidden_dim*2, 
                                    self.output_dim, 
                                    self.args.pooler_activation_fn,
                                    self.args.pooler_dropout)
        else:
            self.classification_head = ClassificationHead(
                input_dim=self.args.encoder_embed_dim,
                inner_dim=self.args.encoder_embed_dim,
                num_classes=self.output_dim,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        self.apply(init_bert_params)
        self.load_pretrained_weights(path=self.pretrain_path)

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
        src_coord,
        src_edge_type,
        gas_id=None,
        gas_attr=None,
        pressure=None,
        temperature=None,
        return_repr=False,
        return_atomic_reprs=False,
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
        cls_repr = encoder_rep[:, 0, :]  # CLS token repr
        all_repr = encoder_rep[:, :, :]  # all token repr

        filtered_tensors = []
        filtered_coords = []
        for tokens, coord in zip(src_tokens, src_coord):
            filtered_tensor = tokens[(tokens != 0) & (tokens != 1) & (tokens != 2)] # filter out BOS(0), EOS(1), PAD(2)
            filtered_coord = coord[(tokens != 0) & (tokens != 1) & (tokens != 2)]
            filtered_tensors.append(filtered_tensor)
            filtered_coords.append(filtered_coord)

        lengths = [len(filtered_tensor) for filtered_tensor in filtered_tensors] # Compute the lengths of the filtered tensors
        if return_repr and return_atomic_reprs:
            cls_atomic_reprs = [] 
            atomic_symbols = []
            for i in range(len(all_repr)):
                atomic_reprs = encoder_rep[i, 1:lengths[i]+1, :]
                atomic_symbol = []
                for atomic_num in filtered_tensors[i]:
                    atomic_symbol.append(self.dictionary.symbols[atomic_num])
                atomic_symbols.append(atomic_symbol)
                cls_atomic_reprs.append(atomic_reprs)
            return {'cls_repr': cls_repr, 
                    'atomic_symbol': atomic_symbols, 
                    'atomic_coords': filtered_coords, 
                    'atomic_reprs': cls_atomic_reprs}        
        if return_repr and not return_atomic_reprs:
            return {'cls_repr': cls_repr}  

        if self.data_type == 'mof':
            gas_embed = self.gas_embed(gas_id, gas_attr) # shape of gas_embed is [batch_size, gas_dim*2]
            env_embed = self.env_embed(pressure, temperature) # shape of gas_embed is [batch_size, env_dim*3]
            rep = torch.cat([cls_repr, gas_embed, env_embed], dim=-1)
            logits = self.classifier(rep)
        else:
            logits = self.classification_head(cls_repr)

        return logits
    
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
            if k != 'smile':
                batch[k] = v
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

def protein_architecture():
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

def crystal_architecture():
    args = argparse.ArgumentParser()
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", "linear")
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    return args

def mof_architecture():
    args = argparse.ArgumentParser()
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.2)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", "linear")
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.gas_attr_input_dim = getattr(args, "gas_attr_input_dim", 6)
    args.hidden_dim = getattr(args, "hidden_dim", 128)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.bins = getattr(args, "bins", 32)
    return args

def oled_architecture():
    args = argparse.ArgumentParser()
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", "linear")
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    return args

class MMI_Model(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, config, ctc_output_size, label_output_size, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2, auxnum_labels=2):
        #super(BertPreTrainedModel, self).__init__()
        super(MMI_Model, self).__init__()
        self.num_labels = num_labels
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base",output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        # self.wav2vec2 = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv",output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        # self.wav2vec2 = HubertModel.from_pretrained("facebook/hubert-base-ls960",output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        self.wav2vec2.feature_extractor._freeze_parameters()
        #self.trans_matrix = torch.zeros(num_labels, auxnum_labels)
        # self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(768, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(768, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.img2txt_attention = BertCrossEncoder(config, layer_num2)
        self.txt2txt_attention = BertCrossEncoder(config, layer_num3)
        # self.txt2fbank_attention = BertCrossEncoder(config, 1)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Sequential(OrderedDict([
        
          ('linear3', nn.Linear(config.hidden_size * 2, label_output_size))
          ]))
        

        self.dropout_audio_input = nn.Dropout(0.1)

        # self.audio_encoder = TransformerEncoder(config_audio)
        self.ctc_linear = nn.Linear(768, ctc_output_size)

        
        self.downsample_final = nn.Linear(768*2, 768)


        self.weights = nn.Parameter(torch.zeros(13))

        self.fuse_type = 'max'

        if self.fuse_type == 'att':
            self.output_attention_audio = nn.Sequential(
                nn.Linear(768, 768 // 2),
                ActivateFun("gelu"),
                nn.Linear(768 // 2, 1)
            )
            self.output_attention_multimodal = nn.Sequential(
                nn.Linear(768*2, 768*2 // 2),
                ActivateFun("gelu"),
                nn.Linear(768*2 // 2, 1)
            )
        

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.

    def _ctc_loss(self, logits, labels, input_lengths, attention_mask=None):

        loss = None
        if labels is not None:

            if attention_mask is not None:
                input_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).type(torch.IntTensor)


            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="sum",
                    zero_infinity=False,
                    )

        return loss

    def _cls_loss(self, logits, cls_labels): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss

    def _weighted_sum(self, feature, normalize):

        stacked_feature = torch.stack(feature, dim=0)

        if normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],))

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(13, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def forward(self, text_output, bert_attention_mask, audio_input, audio_length, ctc_labels, emotion_labels, augmentation = False):

        # text_output = self.bert(input_ids,attention_mask=bert_attention_mask,token_type_ids=bert_segment_ids)
        # text_output = self.dropout(text_output[0])
        text_output = self.dropout(text_output)
        # audio_output_wav2vec2_all = self.wav2vec2(audio_input) #only in average
        # audio_output_wav2vec2 = audio_output_wav2vec2_all[0] #only in average
        audio_output_wav2vec2 = self.wav2vec2(audio_input)[0] #imp
        audio_attention_mask, fbank_attention_mask, wav2vec2_attention_mask, input_lengths = None, None, None, None

        audio_attention_mask = create_mask(audio_input.shape[0],audio_input.shape[1],audio_length)

        input_lengths = self.wav2vec2._get_feat_extract_output_lengths(audio_attention_mask.sum(-1)).type(torch.IntTensor)
        wav2vec2_attention_mask = create_mask(audio_output_wav2vec2.shape[0],audio_output_wav2vec2.shape[1],input_lengths)

        wav2vec2_attention_mask = wav2vec2_attention_mask.cuda()

        #-----------------------------------------------------------------------------------------------------------#

        audio_output_dropout = self.dropout_audio_input(audio_output_wav2vec2)
        logits_ctc = self.ctc_linear(audio_output_dropout)

        extended_txt_mask = bert_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        main_addon_sequence_encoder = self.self_attention_v2(text_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1]

        wav2vec2_attention_mask_back = wav2vec2_attention_mask.clone()
        # subsample the frames to 1/4th of the number
        audio_output = audio_output_wav2vec2.clone()
        # audio_output, wav2vec2_attention_mask = self.conv2d_subsample(audio_output_wav2vec2,wav2vec2_attention_mask.unsqueeze(1)) # remove _2

        converted_vis_embed_map = self.vismap2text(audio_output)

        #--------------------applying txt2img attention mechanism to obtain image-based text representations----------------------------#

        img_mask = wav2vec2_attention_mask.squeeze(1).clone()
        # calculate extended_img_mask required for cross-attention
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim


        #----------------------apply img2txt attention mechanism to obtain multimodal-based text representations-------------------------#

        # project audio embeddings to a smaller space || left part of the image
        converted_vis_embed_map_v2 = self.vismap2text_v2(audio_output)

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]  # self.batch_size * audio_length * hidden_dim

                                            #----------------------------------#

        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        #----------------------------------------------------------------------------------------------------------------------------------#

        #---------------------------------------apply visual gate and get final representations---------------------------------------------#

        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)
        # final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)

        # final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed, gated_converted_att_vis_embed_fbank), dim=-1)
        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)


        audio_output_pool = audio_output_wav2vec2.clone() #change _2
        # audio_output_pool = audio_output.clone()

        multimodal_output = final_output.clone()

        text_output_2 = text_output.clone()

        if self.fuse_type == 'mean':
            if audio_attention_mask is None:
                classification_feats_audio = torch.mean(audio_output_wav2vec2, dim=1)
            else:
                padding_mask = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1], audio_attention_mask)
                padding_mask = padding_mask.to(audio_output_wav2vec2.device)
                audio_output_pool[~padding_mask] = 0.0 #mean
                classification_feats_audio = audio_output_pool.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1) #mean
        elif self.fuse_type == 'max':
            padding_mask = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1], audio_attention_mask)
            padding_mask = padding_mask.to(audio_output_wav2vec2.device)
            audio_output_pool[~padding_mask] = -9999.9999 #max
            classification_feats_audio,_ = torch.max(audio_output_pool,dim = 1) #max
        elif self.fuse_type == 'att':
            text_image_mask = wav2vec2_attention_mask_back.permute(1, 0).contiguous()
            # text_image_mask = wav2vec2_attention_mask.squeeze(1).permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:audio_output_pool.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention_audio(audio_output_pool)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)
            classification_feats_audio = (text_image_alpha.unsqueeze(-1) * audio_output_pool).sum(dim=1)
        elif self.fuse_type == 'stats':
            classification_feats_audio = torch.cat((torch.mean(audio_output_pool,dim=1),torch.std(audio_output_pool,dim=1)), dim=-1) #768*2


        if self.fuse_type == 'mean':
            padding_mask_text = bert_attention_mask > 0
            multimodal_output[~padding_mask_text] = 0.0 #mean
            classification_feats_multimodal = multimodal_output.sum(dim=1) / padding_mask_text.sum(dim=1).view(-1, 1) #mean
            # classification_feats_multimodal = torch.mean(final_output, dim=1)
        elif self.fuse_type == 'max':
            padding_mask_text = bert_attention_mask > 0
            multimodal_output[~padding_mask_text] = -9999.9999 #max
            classification_feats_multimodal,_ = torch.max(multimodal_output,dim = 1) #max
        elif self.fuse_type == 'att':
            multimodal_mask = bert_attention_mask.permute(1, 0).contiguous()
            multimodal_mask = multimodal_mask[0:multimodal_output.size(1)]
            multimodal_mask = multimodal_mask.permute(1, 0).contiguous()

            multimodal_alpha = self.output_attention_multimodal(multimodal_output)
            multimodal_alpha = multimodal_alpha.squeeze(-1).masked_fill(multimodal_mask == 0, -1e9)
            multimodal_alpha = torch.softmax(multimodal_alpha, dim=-1)
            classification_feats_multimodal = (multimodal_alpha.unsqueeze(-1) * multimodal_output).sum(dim=1)
        elif self.fuse_type == 'stats':
            classification_feats_multimodal = torch.cat((torch.mean(multimodal_output,dim=1),torch.std(multimodal_output,dim=1)), dim=-1)

        classification_feats_multimodal = self.downsample_final(classification_feats_multimodal)
        final_output = torch.cat((classification_feats_audio, classification_feats_multimodal), dim=-1)
        classification_feats_pooled = self.classifier(final_output)

        #------------------------------------------------------------------------------------------------------------------------------------#

        #------------------------------------------------------calculate losses---------------------------------------------------------------#

        loss = None
        loss_ctc = None
        loss_cls = None
        if not augmentation:
            loss_ctc = self._ctc_loss(logits_ctc, ctc_labels, input_lengths, audio_attention_mask) #ctc loss
            loss_cls = self._cls_loss(classification_feats_pooled, emotion_labels) #cls loss

        return classification_feats_pooled, final_output, loss_cls, loss_ctc


class TestModel_1(nn.Module):
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
        config=  BertConfig('/workspace1/longnd38/unimol/Uni-Mol/unimol_tools/unimol_tools/config/bert_cfg.json')
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
        
        if data_type == 'mof':
            self.min_max_key = {
                'pressure': [-4.0, 6.0],      # transoformed pressure in log10(P)
                'temperature': [100, 400.0],  
             }
            self.gas_embed = GasModel(self.args.gas_attr_input_dim, self.args.hidden_dim)
            self.env_embed = EnvModel(self.args.hidden_dim, self.args.bins, self.min_max_key)
            self.classifier = ClassificationHead(self.args.encoder_embed_dim+self.args.hidden_dim*5, 
                                    self.args.hidden_dim*2, 
                                    self.output_dim, 
                                    self.args.pooler_activation_fn,
                                    self.args.pooler_dropout)
        else:
            self.classification_head = ClassificationHead(
                input_dim=896,
                inner_dim=self.args.encoder_embed_dim,
                num_classes=self.output_dim,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        self.apply(init_bert_params)
        self.load_pretrained_weights(path=self.pretrain_path)

        self.bert= AutoModel.from_pretrained("/workspace1/longnd38/unimol/chemberta")

        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(512, 384)
        self.vismap2text_v2 = nn.Linear(512, 384)
        self.txt2img_attention = BertCrossEncoder(config, 1)
        self.img2txt_attention = BertCrossEncoder(config, 1)
        self.txt2txt_attention = BertCrossEncoder(config, 1)
        self.gate = nn.Linear(384 * 2, 384)

        self.dropout_audio_input = nn.Dropout(0.1)
        self.downsample_final = nn.Linear(384*2, 384)
        self.fuse_type = 'mean'
        self.infonce= InfoNCE(512, 384)

        # weights for weighted sum
        self.audio_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.text_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1.data.fill_(1)
        self.text_weight_1.data.fill_(1)
        self.bias.data.fill_(0)

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
        src_coord,
        src_edge_type,
        input_ids,
        attention_mask,
        gas_id=None,
        gas_attr=None,
        pressure=None,
        temperature=None,
        return_repr=False,
        return_atomic_reprs=False,
        return_ct_loss=False,
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
            ct_loss= self.infonce(all_repr, out_bert)
        # ct_loss= self.infonce(all_repr, out_bert)

        #  B, L, 384
        """for bert"""
        text_output = self.dropout(out_bert)
        extended_txt_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        main_addon_sequence_encoder = self.self_attention_v2(text_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1]

        img_mask = padding_mask.clone() # cần check lại # .squeeze(1)

        audio_output= all_repr.clone()

 
        """for graph"""
        # from 512 to 384
        converted_vis_embed_map = self.vismap2text(audio_output)

         # .squeeze(1)
        # calculate extended_img_mask required for cross-attention
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

            ### query: text, key, value: graph
        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]
            ### query: graph, key, value: text
        converted_vis_embed_map_v2 = self.vismap2text_v2(audio_output)
        """bert"""
                   ### query: graph, key, value: text

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]

        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]    

        ###                                    
        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)

        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)

        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)

        audio_output_pool = audio_output.clone()

        multimodal_output = final_output.clone()
        eps = 1e-8

        if self.fuse_type=='mean':
            if padding_mask is None:
                classification_feats_audio = torch.mean(audio_output, dim=1)
            else:
                padding_mask = padding_mask.to(all_repr.device)
                audio_output_pool[~padding_mask] = 0.0
                # if padding_mask.sum(dim=1).view(-1, 1)==0:
                #     print("padding_mask.sum(dim=1).view(-1, 1): ",padding_mask.sum(dim=1).view(-1, 1))
                classification_feats_audio = audio_output_pool.sum(dim=1) / (padding_mask.sum(dim=1).view(-1, 1)+ eps)
        elif self.fuse_type=='max':
            padding_mask = padding_mask.to(all_repr.device)
            audio_output_pool[~padding_mask] = -9999.9999
            classification_feats_audio,_ = torch.max(audio_output_pool,dim = 1)
        elif self.fuse_type=='stats':
            classification_feats_audio = torch.cat((torch.mean(audio_output_pool,dim=1),torch.std(audio_output_pool,dim=1)), dim=-1)

        if self.fuse_type=='mean':
            padding_mask_text = attention_mask > 0
            multimodal_output[~padding_mask_text] = 0.0
            classification_feats_multimodal = multimodal_output.sum(dim=1) / (padding_mask_text.sum(dim=1).view(-1, 1)+ eps)

        elif self.fuse_type=='max':
            padding_mask_text = attention_mask > 0
            multimodal_output[~padding_mask_text] = -9999.9999
            classification_feats_multimodal,_ = torch.max(multimodal_output,dim = 1)
        elif self.fuse_type=='stats':
            classification_feats_multimodal = torch.cat((torch.mean(multimodal_output,dim=1),torch.std(multimodal_output,dim=1)), dim=-1)
        
        classification_feats_multimodal = self.downsample_final(classification_feats_multimodal)
        # print("classification_feats_multimodal: ",classification_feats_multimodal)
        # print("classification_feats_audio: ",classification_feats_audio)
        final_output = torch.cat((classification_feats_audio, classification_feats_multimodal), dim=-1)
        # print("final_output: ",final_output.shape)
        logits = self.classification_head(final_output)
        # print("logits: ",logits)

        if return_ct_loss:
            return logits, ct_loss
        return logits
    
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