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
from ..utils import logger
from transformers import AutoTokenizer
import warnings
import pickle
warnings.filterwarnings("ignore")
using_smiles = True

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
        print("Target cols: {}".format(self.target_cols))
        self.multiclass_cnt = params.get('multiclass_cnt', None)
        self.cache_dir = params.get('cache_dir', None)
        self.ss_method = params.get('target_normalize', 'none')
        # self.cache_dir= cache_dir
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
        self.data['target_scaler'] = TargetScaler(self.ss_method, self.task, self.save_path)
        if self.task == 'regression': 
            target = np.array(self.data['raw_target']).reshape(-1,1).astype(np.float32)
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path)
                self.data['target'] = self.data['target_scaler'].transform(target)
            else:
                self.data['target'] = target
        elif self.task == 'classification':
            target = np.array(self.data['raw_target']).reshape(-1,1).astype(np.int32)
            self.data['target'] = target
        elif self.task =='multiclass':
            target = np.array(self.data['raw_target']).reshape(-1,1).astype(np.int32)
            self.data['target'] = target
            if not self.is_train:
                self.data['multiclass_cnt'] = self.multiclass_cnt 

        # focus
        elif self.task == 'multilabel_regression':
            target = np.array(self.data['raw_target']).reshape(-1,self.data['num_classes']).astype(np.float32)
            if self.is_train:
                self.data['target_scaler'].fit(target, self.save_path)
                self.data['target'] = self.data['target_scaler'].transform(target)                
            else:
                self.data['target'] = target
        elif self.task == 'multilabel_classification':
            target = np.array(self.data['raw_target']).reshape(-1,self.data['num_classes']).astype(np.int32)
            self.data['target'] = target
        elif self.task == 'repr':
            self.data['target'] = self.data['raw_target']
        else:
            raise ValueError('Unknown task: {}'.format(self.task))

        if self.cache_dir is not None:
            if os.path.exists(self.cache_dir):
                with open(self.cache_dir, 'rb') as f:
                    no_h_list= pickle.load(f)
                    print("Load data from cache...")
                self.data['unimol_input']= no_h_list
            else:

                if 'atoms' in self.data and 'coordinates' in self.data:
                    no_h_list = ConformerGen(**params).transform_raw(self.data['atoms'], self.data['coordinates'])
                else:
                    smiles_list = self.data["smiles"]                  
                    no_h_list = ConformerGen(**params).transform(smiles_list)

                for idx in range(len(no_h_list)):
                    no_h_list[idx]['smile']= self.data['smiles'][idx]

                with open(self.cache_dir, 'wb') as f:
                    pickle.dump( no_h_list, f)
                    print("Save data to cache...")
                self.data['unimol_input']= no_h_list
        else:
            if 'atoms' in self.data and 'coordinates' in self.data:
                no_h_list = ConformerGen(**params).transform_raw(self.data['atoms'], self.data['coordinates'])
            else:
                smiles_list = self.data["smiles"]                  
                no_h_list = ConformerGen(**params).transform(smiles_list)

            for idx in range(len(no_h_list)):
                no_h_list[idx]['smile']= self.data['smiles'][idx]
            self.data['unimol_input']= no_h_list

    