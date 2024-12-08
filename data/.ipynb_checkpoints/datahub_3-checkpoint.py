# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
from typing import List, Optional
from collections import defaultdict
from .datareader import MolDataReader
from .datascaler import TargetScaler
from .conformer import ConformerGen
from ..utils import logger, get_lds_kernel_window
from scipy.ndimage import convolve1d
import warnings
warnings.filterwarnings("ignore")
using_smiles = True
import pickle
# from fairseq.data.lru_cache_dataset import LRUCacheDataset
from fairseq.data.append_token_dataset import AppendTokenDataset
from functools import lru_cache, reduce
import logging
import os
import numpy as np
from numpy.core.fromnumeric import sort
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    NumSamplesDataset,
    NumelDataset,
    data_utils,
    LeftPadDataset,
    BaseWrapperDataset,
    RawLabelDataset,
)
from fairseq.data.shorten_dataset import TruncateDataset, maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from dataclasses import dataclass, field
from typing import Optional, List, Any
from omegaconf import II
from fairseq.data.indexed_dataset import (
    MMapIndexedDataset,
    get_available_dataset_impl,
    make_dataset,
    infer_dataset_impl,
)
from fairseq.data.molecule.indexed_dataset import MolMMapIndexedDataset
from fairseq.data.molecule.indexed_dataset import make_dataset as make_graph_dataset
from fairseq.data.molecule.molecule import Tensor2Data
from fairseq.tasks.doublemodel import NoiseOrderedDataset, StripTokenDatasetSizes
import argparse
from tqdm import tqdm
def lds_config():
    args = argparse.ArgumentParser()
    args.lds_kernel = getattr(args, "lds_kernel", "gaussian")
    args.lds_ks = getattr(args, "lds_ks", 5)
    args.lds_sigma = getattr(args, "lds_sigma", 2)
    return args 

def anomaly_clean_regression( data):
    """
    Performs anomaly cleaning specifically for regression tasks using a 3-sigma threshold.

    :param data: (DataFrame) The dataset to be cleaned.
    :param target_cols: (list) The list of target columns to consider for cleaning.

    :return: (DataFrame) The cleaned dataset after applying the 3-sigma rule.
    """
    _mean, _std = data.mean(), data.std()
    data = data[(data > _mean - 3 * _std) & (data < _mean + 3 * _std)]
    return data 

def calculate_weights(regression_value, reweight= 'sqrt_inv', max_bin=200):
    """Calculates weights based on the distribution of regression values.

    Args:
        regression_value (np.ndarray): Array of regression values.
        max_bin (int, optional): Maximum number of bins for the histogram. Defaults to 200.

    Returns:
        np.ndarray: Array of weights for each data point.
    """
    if isinstance(regression_value, tuple):
        lds= regression_value[1]
        regression_value= np.array(regression_value[0])
        

    regression_value_org= copy.deepcopy(regression_value)
    regression_value= anomaly_clean_regression(regression_value)
    value_range = np.max(regression_value) - np.min(regression_value)
    bin_width = value_range / max_bin

    value_dict = {k: 0 for k in range(max_bin + 1)}
    for value in tqdm(regression_value):
        
        bin_index= int((value-np.min(regression_value))//bin_width)
        value_dict[bin_index]+=1
    
    value_dict_no = {k: v for k, v in value_dict.items() if v!=0}
    min_index= np.min(list(value_dict_no.keys()))
    max_index= np.max(list(value_dict_no.keys()))

    if reweight == 'sqrt_inv':
        logger.info('Using SQRT Inverse')
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()} 

    num_per_label= []

    for value in regression_value_org:
        bin_index= int((value-np.min(regression_value))//bin_width)
        if bin_index< min_index:
            bin_index= min_index
        if bin_index> max_index:
            bin_index= max_index
        num_per_label.append(value_dict[bin_index])

    if lds:
        lds_cfg= lds_config()
        lds_kernel_window = get_lds_kernel_window(lds_cfg.lds_kernel, lds_cfg.lds_ks, lds_cfg.lds_sigma)
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        
        list_keys= list(value_dict.keys())
        assert len(list_keys) == len(smoothed_value)
        smoothed_value = {list_keys[i]: smoothed_value[i] for i in range(len(list_keys))}
        smooth_dict_no = {k: v for k, v in smoothed_value.items() if v!=0}
        min_index= np.min(list(smooth_dict_no.keys()))
        max_index= np.max(list(smooth_dict_no.keys()))
        num_per_label = []
        for value in regression_value_org:
            bin_index= int((value-np.min(regression_value))//bin_width)
            if bin_index< min_index:
                bin_index= min_index
            if bin_index> max_index:
                bin_index= max_index
            num_per_label.append(smoothed_value[bin_index])
        


    weights= [np.float32(1/x) for x in num_per_label]
    scalings= len(weights)/np.sum(weights)
    weights= [scalings*x for x in weights]

    return weights


def optimize_weighting_parallel(data, lds= False):
    """Optimizes weights based on the distribution of regression values in the data using multiprocessing.

    Args:
        data (dict): Dictionary containing target data.
    """

    logger.info('Using all weight with multiprocessing')
    if lds:
        logger.info('Using label distribution smoothing')
    num_cores = 18  # Use available CPU cores by default

    with mp.Pool(processes=num_cores) as pool:
        # Prepare tasks for the pool (function and arguments for each regression value)
        tasks = [(data[:, idx].reshape(-1), lds) for idx in range(18)]


        # Use tqdm with pool.imap_unordered for progress bar (avoid blocking the main thread)
        weights_list = []
        with tqdm(total=len(tasks)) as pbar:
            for result in pool.map(calculate_weights, tasks):
                weights_list.append(result)
                pbar.update(1)  # Update progress bar for each completed task

    return weights_list


def optimize_weighting_parallel_2(data, lds= False):
    """Optimizes weights based on the distribution of regression values in the data using multiprocessing.

    Args:
        data (dict): Dictionary containing target data.
    """

    logger.info('Using 1 weight with multiprocessing')
    if lds:
        logger.info('Using label distribution smoothing')
    num_cores = 1  # Use available CPU cores by default

    with mp.Pool(processes=num_cores) as pool:
        # Prepare tasks for the pool (function and arguments for each regression value)
        tasks = [(data[:, idx].reshape(-1), lds) for idx in range(1)]


        # Use tqdm with pool.imap_unordered for progress bar (avoid blocking the main thread)
        weights_list = []
        with tqdm(total=len(tasks)) as pbar:
            for result in pool.map(calculate_weights, tasks):
                weights_list.append(result)
                pbar.update(1)  # Update progress bar for each completed task

    return weights_list[0]

def get_dataset(path, train= False):
    def get_path(key, split):
        return os.path.join(path, split)
    data_dictionary= Dictionary.load(os.path.join(path, "dict.txt"))
    print(
            "[input] Dictionary {}: {} types.".format(
                os.path.join(path, "input0",), len(data_dictionary)
            )
    )
    
    data_dictionary.add_symbol("[MASK]")
    max_positions= 512
    if train:
        prefix = get_path("input0", "train")
    else:
        
        prefix = get_path("input0", "valid")
    if not MMapIndexedDataset.exists(prefix):
        raise FileNotFoundError("Graph data {} not found.".format(prefix))

    src_dataset = make_dataset(prefix, impl="mmap")
    assert src_dataset is not None

    src_dataset = AppendTokenDataset(
        TruncateDataset(
            StripTokenDatasetSizes(src_dataset, data_dictionary.eos()),
            max_positions - 1,
        ),
        data_dictionary.eos(),
    )
    dataset = {
        "id": IdDataset(),
        "net_input": {
            "src_tokens": LeftPadDataset(src_dataset, pad_idx=data_dictionary.pad()),
            "src_lengths": NumelDataset(src_dataset),
        },
        "nsentences": NumSamplesDataset(),
        "ntokens": NumelDataset(src_dataset, reduce=True),
    }
    return dataset

class DataHub_3(object):
    """
    The DataHub class is responsible for storing and preprocessing data for machine learning tasks.
    It initializes with configuration options to handle different types of tasks such as regression, 
    classification, and others. It also supports data scaling and handling molecular data.
    """
    def __init__(self, data=None, is_train=True, save_path=None, data_train= None, get_train= True,**params):
        """
        Initializes the DataHub instance with data and configuration for the ML task.

        :param data: Initial dataset to be processed.
        :param is_train: (bool) Indicates if the DataHub is being used for training.
        :param save_path: (str) Path to save any necessary files, like scalers.
        :param params: Additional parameters for data preprocessing and model configuration.
        """
        self.data = data
        self.is_train = is_train
        self.save_path = save_path
        self.task = params.get('task', None)
        self.target_cols = params.get('target_cols', None)
        print("Target cols: {}".format(self.target_cols))
        self.multiclass_cnt = params.get('multiclass_cnt', None)
        self.cache_dir_train = params.get('cache_dir_train', None)
        self.cache_dir_test = params.get('cache_dir_test', None)
        self.ss_method = params.get('target_normalize', 'none')
        self.all_weight=  params.get('all_weight', False)
        self.raw_data= params.get('raw_data', False)
        self.lds= params.get('lds', False)
        self.data_train= data_train
        self.get_train= get_train
        self._init_data(**params)
    
    def _init_data(self, **params):
        """
        Initializes and preprocesses the data based on the task and parameters provided.

        This method handles reading raw data, scaling targets, and transforming data for use with 
        molecular inputs. It tailors the preprocessing steps based on the task type, such as regression 
        or classification.

        :param params: Additional parameters for data processing.
        :raises ValueError: If the task type is unknown.
        """
        self.data = MolDataReader().read_data(self.data, self.is_train, **params)
        self.raw_data_target= pd.read_csv(self.raw_data)[self.target_cols].values
        self.data['target_scaler'] = TargetScaler(self.ss_method, self.task, self.save_path)
        if self.task == 'regression':
            target = np.array(self.data['raw_target']).reshape(-1, 1).astype(np.float32)
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path)
                self.data['target'] = self.data['target_scaler'].transform(target)
                
            else:
                self.data['target'] = target
                # self.data['target'] = self.data['target_scaler'].transform(target)
                
        elif self.task == 'classification':
            target = np.array(self.data['raw_target']).reshape(-1, 1).astype(np.int32)
            self.data['target'] = target
        elif self.task == 'multiclass':
            target = np.array(self.data['raw_target']).reshape(-1, 1).astype(np.int32)
            self.data['target'] = target
            if not self.is_train:
                self.data['multiclass_cnt'] = self.multiclass_cnt

                # focus
        elif self.task == 'multilabel_regression':
            target = np.array(self.data['raw_target']).reshape(-1, self.data['num_classes']).astype(np.float32)
            if self.is_train:
                self.data['target_scaler'].fit(self.raw_data_target, self.save_path)
                self.data['target'] = self.data['target_scaler'].transform(target)
                logger.info('Creating target scaler from raw data...')
            else:
                # self.data['target'] = target
                self.data['target'] = self.data['target_scaler'].transform(target)
                logger.info('Using target scaler ...')
        elif self.task == 'multilabel_classification':
            target = np.array(self.data['raw_target']).reshape(-1, self.data['num_classes']).astype(np.int32)
            self.data['target'] = target
        elif self.task == 'repr':
            self.data['target'] = self.data['raw_target']
        else:
            raise ValueError('Unknown task: {}'.format(self.task))
        # weights
        if self.all_weight:
            # logger.info('Using all weight')
            weights_list= np.array(optimize_weighting_parallel(self.data['target'], self.lds))
            new_weights_list = []
            for i in range(weights_list.shape[1]):
                new_sublist = []
                for weight in weights_list:
                    new_sublist.append(weight[i])
                new_weights_list.append(new_sublist)
            self.data['weights']= np.array(new_weights_list)
            
        else:
            logger.info('Using 1 weight')
            self.data['weights']= np.array(optimize_weighting_parallel_2(self.data['target'], self.lds))           
        #
        if self.is_train:
            cache_dir= self.cache_dir_train
        else:
            cache_dir = self.cache_dir_test
        if cache_dir is not None:
            if os.path.exists(cache_dir):
                with open(cache_dir, 'rb') as f:
                    no_h_list = pickle.load(f)
                    logger.info("Load data from cache...")
                    
                for idx in range(len(no_h_list)):
                    no_h_list[idx]['weights'] = self.data['weights'][idx]
                self.data['unimol_input'] = no_h_list
                
            else:

                if 'atoms' in self.data and 'coordinates' in self.data:
                    no_h_list = ConformerGen(**params).transform_raw(self.data['atoms'], self.data['coordinates'])
                else:
                    smiles_list = self.data["smiles"]
                    no_h_list = ConformerGen(**params).transform(smiles_list)

                for idx in range(len(no_h_list)):
                    no_h_list[idx]['smile'] = self.data['smiles'][idx]

                with open(cache_dir, 'wb') as f:
                    pickle.dump(no_h_list, f)
                    logger.info("Save data to cache...")
                self.data['unimol_input'] = no_h_list
        else:
            if 'atoms' in self.data and 'coordinates' in self.data:
                no_h_list = ConformerGen(**params).transform_raw(self.data['atoms'], self.data['coordinates'])
            else:
                smiles_list = self.data["smiles"]
                no_h_list = ConformerGen(**params).transform(smiles_list)

            for idx in range(len(no_h_list)):
                no_h_list[idx]['smile'] = self.data['smiles'][idx]
            self.data['unimol_input'] = no_h_list

        # data for bert
        if self.data_train is not None:
            data_bert= get_dataset(self.data_train, self.get_train)
            self.data['net_input']=[data_bert['net_input']["src_tokens"][i] for i in range(len(data_bert['net_input']["src_tokens"]))]
            print(len(self.data['net_input']))


        assert len(self.data['net_input'])==len(no_h_list), "size data bert and graph must match"
        new_no_h_list= []
        for idx in range(len(no_h_list)):
            new_idx= no_h_list[idx]
            new_idx.update({'net_input':self.data['net_input'][idx]})
            new_no_h_list.append(new_idx)
        self.data['unimol_input'] = new_no_h_list
        

            # self.data['input_ids'] = [id['input_ids'] for id in input_bert]
            # self.data['attention_mask'] = [id['attention_mask'] for id in input_bert]
