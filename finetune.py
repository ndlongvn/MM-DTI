from train import MolTrain
from predict import MolPredict
from tasks import random_scaffold_split, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, precision_recall_curve
import csv
import os


data_path= 'dataset/molecunet/regression/refined_FreeSolv.csv' # path to the data
col_name= ['measured'] # name of the target column
seed=42 # seed for training
batch_size=32 # batch size
epoch=1 # number of epochs
learning_rate=1e-4 # learning rate
using_scaler= True # whether to use scaler for regression
fds_num=30 # number of bukcet for FDS
use_weight= True # whether to use weight for ConR loss

result_file= "result.csv" # path to the result file

if not os.path.exists(os.path.dirname(result_file)) and os.path.dirname(result_file) != '':
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

test_rmse= []
for idx in range(5):
    # split data base on split type
    train_dataset, valid_dataset, test_dataset= random_scaffold_split(data_path, random_seed= idx, ratio_test= 0.1, ration_valid= 0.1)

    # save data to csv file
    train_dataset.to_csv('train.csv', index=False)
    valid_dataset.to_csv('val.csv', index=False)
    test_dataset.to_csv('test.csv', index=False)


    clf = MolTrain(task='regression', # ['classification', 'regression', 'multilabel_classification, multilabel_regression']
                    data_type='molecule',
                    epochs=epoch,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    early_stopping=20,
                    metrics='mse',
                    smiles_col='smiles', # column name of smiles
                    save_path= './exp_seed_{}'.format(idx), # path to save the model
                    target_cols=col_name,
                    use_cuda=True,
                    model_name= 'mm_model', # name of model
                    using_infonce= True, # using infonce loss
                    using_ct= True,# using ct loss
                    raw_data= 'train.csv', # path to training data
                    use_weight= use_weight,
                    all_weight= False, # using all weight for multilabel-regression
                    fds= True, # using FDS
                    seed= seed, # seed for training
                    cache_dir_train=None, # path to cache file train for reusing
                    cache_dir_test=None, # path to cache file test for reusing
                    target_anomaly_check= 'filter', # using filter for anomaly check
                    using_scaler= using_scaler, # using scaler for regression
                    fds_num= fds_num,  # number of bucket for FDS
                    fds_raw_path= 'train.csv', # path to training data
                    fds_col_data= col_name[0], # column name of target for FDS
                    chemberta_dir= 'weights/ChemBERTa', # path to chemberta pretrained model
                    unimol_dir= 'weights/Uni-Mol/mol_pre_all_h_220816.pt', # path to unimol pretrained model
            )

    clf.fit('train.csv', 'val.csv')
    clf= MolPredict(load_model = './exp_seed_{}'.format(idx), cache_dir=None) # 

    data_test = pd.read_csv('test.csv')
    test_pred = clf.predict('test.csv')
    rmse= np.sqrt(mean_squared_error(data_test[clf.config['target_cols']], test_pred))
    test_rmse.append(rmse)

df= pd.DataFrame({
    'seed': list(range(5)),
    'rmse': test_rmse
})
df.to_csv(result_file, index=False)            
                
