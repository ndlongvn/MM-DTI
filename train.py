# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from data import DataHub
from models import NNModel
from tasks import Trainer
from utils import YamlHandler
from utils import logger

class MolTrain(object):
    """A :class:`MolTrain` class is responsible for interface of training process of molecular data."""
    def __init__(self, 
                task='classification',
                data_type='molecule',
                epochs=10,
                learning_rate=1e-4,
                batch_size=16,
                early_stopping=5,
                metrics= "none",
                save_path='./exp',
                remove_hs=False,
                smiles_col='SMILES',
                target_col_prefix='TARGET',
                target_cols=None,
                target_anomaly_check="filter",
                smiles_check="filter",
                target_normalize="auto",
                max_norm=5.0,
                use_cuda=True,
                use_amp=True,
                model_name= 'unimolv1', 
                chemberta_dir="",
                unimol_dir="",
                using_infonce=False,
                using_ct=False,
                cache_dir_train= None,
                 cache_dir_test= None,
                 use_weight= False,
                 all_weight= False,
                 alpha= 1,
                 beta= 0.1,
                 raw_data=None,
                 fds=False,
                 lds=False,
                 seed=42,
                 use_scaler=True,
                 fds_num= 200,
                 fds_raw_path= '',
                 fds_col_data= '',
                 ct_lamda= 1.0,
                 ct_w=0.2,
                **params,
                ):
        """
        Initialize a :class:`MolTrain` class.

        :param task: str, default='classification', currently support [`]classification`, `regression`, `multiclass`, `multilabel_classification`, `multilabel_regression`.
        :param data_type: str, default='molecule', currently support molecule, oled.
        :param epochs: int, default=10, number of epochs to train.
        :param learning_rate: float, default=1e-4, learning rate of optimizer.
        :param batch_size: int, default=16, batch size of training.
        :param early_stopping: int, default=5, early stopping patience.
        :param metrics: str, default='none', metrics to evaluate model performance.

            currently support: 

            - classification: auc, auprc, log_loss, acc, f1_score, mcc, precision, recall, cohen_kappa. 

            - regression: mse, pearsonr, spearmanr, mse, r2.

            - multiclass: log_loss, acc.

            - multilabel_classification: auc, auprc, log_loss, acc, mcc.

            - multilabel_regression: mae, mse, r2.

        :param split: str, default='random', split method of training dataset. currently support: random, scaffold, group, stratified.
        :param split_group_col: str, default='scaffold', column name of group split.
        :param kfold: int, default=5, number of folds for k-fold cross validation.
        :param save_path: str, default='./exp', path to save training results.
        :param remove_hs: bool, default=False, whether to remove hydrogens from molecules.
        :param smiles_col: str, default='SMILES', column name of SMILES.
        :param target_col_prefix: str, default='TARGET', prefix of target column name.
        :param target_anomaly_check: str, default='filter', how to deal with anomaly target values. currently support: filter, none.
        :param smiles_check: str, default='filter', how to deal with invalid SMILES. currently support: filter, none.
        :param target_normalize: str, default='auto', how to normalize target values. 'auto' means we will choose the normalize strategy by automatic. \
            currently support: auto, minmax, standard, robust, log1p, none.
        :param max_norm: float, default=5.0, max norm of gradient clipping.
        :param use_cuda: bool, default=True, whether to use GPU.
        :param use_amp: bool, default=True, whether to use automatic mixed precision.
        :param freeze_layers: str or list, frozen layers by startwith name list. ['encoder', 'gbf'] will freeze all the layers whose name start with 'encoder' or 'gbf'.
        :param freeze_layers_reversed: bool, default=False, inverse selection of frozen layers
        :param params: dict, default=None, other parameters.

        """
        config_path = os.path.join(os.path.dirname(__file__), 'config/default.yaml')
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        config.task = task
        config.data_type = data_type
        config.epochs = epochs
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.patience = early_stopping
        config.metrics = metrics
        config.remove_hs = remove_hs
        config.smiles_col = smiles_col
        config.target_col_prefix = target_col_prefix
        config.target_cols = target_cols
        config.anomaly_clean = target_anomaly_check in ['filter']
        config.smi_strict = smiles_check in ['filter']
        config.target_normalize = target_normalize
        config.max_norm = max_norm
        config.use_cuda = use_cuda
        config.use_amp = use_amp
        config.model_name = model_name
        config.chemberta_dir = chemberta_dir
        config.unimol_dir = unimol_dir
        config.using_ct = using_ct
        config.using_infonce= using_infonce
        config.cache_dir_train= cache_dir_train
        config.cache_dir_test = cache_dir_test
        config.use_weight = use_weight
        config.all_weight= all_weight
        config.alpha= alpha
        config.beta= beta
        config.raw_data= raw_data
        config.fds= fds
        config.lds= lds
        config.seed= seed
        config.use_scaler= use_scaler
        config.fds_num= fds_num
        config.fds_raw_path= fds_raw_path
        config.fds_col_data= fds_col_data if fds_col_data !='' else target_cols[0]
        config.ct_w= ct_w
        config.ct_lamda= ct_lamda
        self.save_path = save_path
        self.config = config


    def fit(self, data_train, data_val):
        """
        Fit the model according to the given training data with multi datasource support, including SMILES csv file and custom coordinate data.

        For example: custom coordinate data.

        .. code-block:: python

            from unimol_tools import MolTrain
            import numpy as np
            custom_data ={'target':np.random.randint(2, size=100),
                        'atoms':[['C','C','H','H','H','H'] for _ in range(100)],
                        'coordinates':[np.random.randn(6,3) for _ in range(100)],
                        }

            clf = MolTrain()
            clf.fit(custom_data)
        """
        self.datahub = DataHub(data = data_train, is_train=True, save_path=self.save_path,  **self.config)
        self.datahub_1= DataHub(data = data_val, is_train=False, save_path=self.save_path,  **self.config)
        self.data_train = self.datahub.data
        self.data_test = self.datahub_1.data
        self.update_and_save_config()
        self.trainer = Trainer(save_path=self.save_path, **self.config)
        self.model = NNModel(self.data_train, self.data_test, self.trainer, **self.config)
        self.model.run()
        scalar = self.data_train['target_scaler']
        y_pred = self.model.cv['pred']
        y_true = np.array(self.data_train['target'])
        metrics = self.trainer.metrics
        if scalar is not None:
            y_pred = scalar.inverse_transform(y_pred)
            y_true = scalar.inverse_transform(y_true)

        if self.config["task"] in ['classification', 'multilabel_classification']:
            threshold = 0.5#metrics.calculate_classification_threshold(y_true, y_pred)
            joblib.dump(threshold, os.path.join(self.save_path, 'threshold.dat'))
        
        self.cv_pred = y_pred
        return

    def update_and_save_config(self):
        """
        Update and save config file.
        """
        self.config['num_classes'] = self.data_train['num_classes']
        self.config['target_cols'] = ','.join(self.data_train['target_cols'])
        if self.config['task'] == 'multiclass':
            self.config['multiclass_cnt'] = self.data_train['multiclass_cnt']

        self.config['split_method'] = f"{self.config['kfold']}fold_{self.config['split']}"
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                logger.info('Create output directory: {}'.format(self.save_path))
                os.makedirs(self.save_path)
            else:
                logger.info('Output directory already exists: {}'.format(self.save_path))
                logger.info('Warning: Overwrite output directory: {}'.format(self.save_path))
            out_path = os.path.join(self.save_path, 'config.yaml')
            self.yamlhandler.write_yaml(data = self.config, out_file_path = out_path)
        return
