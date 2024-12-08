# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    GroupKFold, 
    KFold, 
    StratifiedKFold,
)
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

class Splitter(object):
    """
    The Splitter class is responsible for splitting a dataset into train and test sets 
    based on the specified method.
    """
    def __init__(self, split_method='5fold_random', seed=42):
        """
        Initializes the Splitter with a specified split method and random seed.

        :param split_method: (str) The method for splitting the dataset, in the format 'Nfold_method'. 
                             Defaults to '5fold_random'.
        :param seed: (int) Random seed for reproducibility in random splitting. Defaults to 42.
        """
        self.n_splits, self.method = int(split_method.split('fold')[0]), split_method.split('_')[-1]    # Nfold_xxxx
        self.seed = seed
        self.splitter = self._init_split()

    def _init_split(self):
        """
        Initializes the actual splitter object based on the specified method.

        :return: The initialized splitter object.
        :raises ValueError: If an unknown splitting method is specified.
        """
        if self.method == 'random':
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.method == 'scaffold' or self.method == 'group':
            splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == 'stratified':
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        else:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))

        return splitter

    def split(self, data, target=None, group=None):
        """
        Splits the dataset into train and test sets based on the initialized method.

        :param data: The dataset to be split.
        :param target: (optional) Target labels for stratified splitting. Defaults to None.
        :param group: (optional) Group labels for group-based splitting. Defaults to None.

        :return: An iterator yielding train and test set indices for each fold.
        :raises ValueError: If the splitter method does not support the provided parameters.
        """
        try:
            return self.splitter.split(data, target, group)
        except:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))



# splitter function

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def random_scaffold_split(dataset, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1):

    rng = np.random.RandomState(random_seed)
    if isinstance(dataset, str):
        dataset= pd.read_csv(dataset)
    if isinstance(dataset, pd.DataFrame):
        try:
            smiles_list= dataset['SMILES'].values #if 'SMILES' in data.columns else data['smiles']
        except:
            smiles_list= dataset['smiles'].values
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    idxs= list(scaffolds.keys())
    idxs = rng.permutation(idxs)
    scaffold_sets = [scaffolds[idx] for idx in idxs]

    
    n_total_test = int(ratio_test * len(dataset))
    n_total_valid = int(ration_valid *(1-ratio_test)* len(dataset))
    
    print('Num train: {}, Num val {}, Num test {}'.format(len(smiles_list)-n_total_test-n_total_valid, n_total_valid, n_total_test))
    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    assert len(set(train_idx)) + len(set(test_idx))+ len(set(valid_idx)) == len(smiles_list), 'total not match'
    train_dataset = dataset.iloc[train_idx]
    valid_dataset = dataset.iloc[valid_idx]
    test_dataset = dataset.iloc[test_idx]

    return train_dataset, valid_dataset, test_dataset

def random_scaffold_split(dataset, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1):
    print('Random scaffold split ...........')
    rng = np.random.RandomState(random_seed)
    if isinstance(dataset, str):
        dataset= pd.read_csv(dataset)

    
    try:
        smiles_list= dataset['smiles'].values
        print("Using smiles column")
    except:
        print("Using SMILES column")
        smiles_list= dataset['SMILES'].values
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    idxs= list(scaffolds.keys())
    idxs = rng.permutation(idxs)
    scaffold_sets = [scaffolds[idx] for idx in idxs]

    n_total_valid = int(ration_valid * len(dataset) * (1-ratio_test))
    n_total_test = int(ratio_test * len(dataset))
    print('Num train: {}, Num val {}, Num test {}'.format(len(smiles_list)-n_total_test-n_total_valid, n_total_valid, n_total_test))
    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    assert len(set(train_idx)) + len(set(test_idx))+ len(set(valid_idx)) == len(smiles_list), 'total not match'
    train_dataset = dataset.iloc[train_idx]
    valid_dataset = dataset.iloc[valid_idx]
    test_dataset = dataset.iloc[test_idx]

    return train_dataset, valid_dataset, test_dataset

def random_split(data, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1):
    print('Random split ...........')
    if isinstance(data, str):
        data= pd.read_csv(data)
   
    X_, X_test = train_test_split(data, test_size=ratio_test, random_state=random_seed)
    X_train, X_val = train_test_split(X_, test_size=ration_valid, random_state=random_seed)
    assert len(X_train) + len(X_val) + len(X_test) == len(data)
    # print('train: {}, valid: {}, test: {}'.format(len(X_train), len(X_val), len(X_test)))
    print('Num train: {}, Num val {}, Num test {}'.format(len(X_train), len(X_val), len(X_test)))
    return X_train, X_val, X_test