from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
from typing import List, Optional
from collections import defaultdict
from data.datareader import MolDataReader
from data.datascaler import TargetScaler
from data.conformer import ConformerGen
from utils import logger, get_lds_kernel_window
import argparse
import warnings
import pickle
from tqdm import tqdm
import multiprocessing as mp
from scipy.ndimage import convolve1d
warnings.filterwarnings("ignore")
using_smiles = True

def lds_config():
    args = argparse.ArgumentParser()
    args.lds_kernel = getattr(args, "lds_kernel", "gaussian")
    args.lds_ks = getattr(args, "lds_ks", 9)
    args.lds_sigma = getattr(args, "lds_sigma", 1)
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
        logger.info('Using LDS')
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
    num_cores = 17  # Use available CPU cores by default

    with mp.Pool(processes=num_cores) as pool:
        # Prepare tasks for the pool (function and arguments for each regression value)
        tasks = [(data[:, idx].reshape(-1), lds) for idx in range(num_cores)]


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

class DataHub(object):
    """
    The DataHub class is responsible for storing and preprocessing data for machine learning tasks.
    It initializes with configuration options to handle different types of tasks such as regression, 
    classification, and others. It also supports data scaling and handling molecular data.
    """

    def __init__(self, data=None, is_train=True, save_path=None, **params):
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
        logger.info("Target cols: {}".format(self.target_cols))
        self.multiclass_cnt = params.get('multiclass_cnt', None)
        self.cache_dir_train = params.get('cache_dir_train', None)
        self.cache_dir_test = params.get('cache_dir_test', None)
        self.ss_method = params.get('target_normalize', 'none')
        self.all_weight=  params.get('all_weight', False)
        self.raw_data= params.get('raw_data', False)
        self.lds= params.get('lds', False)
        self.use_scaler= params.get('use_scaler', True)
        self.max_bin= params.get('fds_num', 200)
        self.use_weight= params.get('use_weight', False)
    

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
        
        # scaler
        if self.use_scaler:
            self.data['target_scaler'] = TargetScaler(self.ss_method, self.task, self.save_path)
            if self.task == 'regression':
                target = np.array(self.data['raw_target']).reshape(-1, 1).astype(np.float32)
                if self.is_train:
                    self.data['target_scaler'].fit(self.raw_data_target, self.save_path)
                    self.data['target'] = self.data['target_scaler'].transform(target)
                    logger.info('Creating target scaler from raw data...')
                    
                else:
                    self.data['target'] = self.data['target_scaler'].transform(target)
                    logger.info('Using target scaler ...')
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
        else:
            logger.info('Not use target scaler ...')
            self.data['target_scaler'] = None
            if self.task == 'regression':
                target = np.array(self.data['raw_target']).reshape(-1, 1).astype(np.float32)
                self.data['target'] = target

                    
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
                self.data['target'] = target
            elif self.task == 'multilabel_classification':
                target = np.array(self.data['raw_target']).reshape(-1, self.data['num_classes']).astype(np.int32)
                self.data['target'] = target
            elif self.task == 'repr':
                self.data['target'] = self.data['raw_target']
            else:
                raise ValueError('Unknown task: {}'.format(self.task))
            # weights
        if self.use_weight:
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
        else:
            self.data['weights']= np.ones_like(self.data['target'])  
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
                    no_h_list[idx]['smile'] = self.data['smiles'][idx]
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
            self.data['unimol_input'] = no_h_list