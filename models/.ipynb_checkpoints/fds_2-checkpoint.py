import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import copy
from tqdm import tqdm 
from  sklearn.preprocessing import StandardScaler

from ..utils import calibrate_mean_var, logger
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

class FDS(nn.Module):

    def __init__(self, feature_dim, raw_data, bucket_num=100, bucket_start=0, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.raw_data= pd.read_csv(raw_data).loc[:, '3CLPro_pocket1'].values
        self.org_data= pd.read_csv("/workspace1/longnd38/unimol/Uni-Mol/unimol_tools/data/train_mocop.csv").loc[:, '3CLPro_pocket1'].values
        # create bucket for set data
        regression_value= copy.deepcopy(self.raw_data)
        std= StandardScaler()
        std.fit(self.org_data.reshape(-1, 1))
        regression_value= std.transform(regression_value.reshape(-1, 1)).reshape(-1)
        regression_value= anomaly_clean_regression(regression_value)
        value_range = np.max(regression_value) - np.min(regression_value)
        # bin_width = value_range / (bucket_num- 1)
        bin_width = value_range / (bucket_num-1)

        # value_dict = {k: 0 for k in range(bucket_num)}
#         for value in tqdm(regression_value):
            
#             bin_index= int((value-np.min(regression_value))//bin_width)
#             value_dict[bin_index]+=1
        
        # self.value_dict= value_dict
        self.min_value= np.min(regression_value)
        self.bin_width= bin_width


        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        logger.info(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            logger.info(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return
        
        labels_0= labels[:, 0]
        label_bin= torch.Tensor([int((value-self.min_value)//self.bin_width) for value in labels_0])

        # labels = labels.reshape(labels.shape[0],-1)
        # labels = torch.mean(labels,dim=1).unsqueeze(-1)

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels_0.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(label_bin):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                curr_feats = features[label_bin <= label]
            elif label == self.bucket_num - 1:
                curr_feats = features[label_bin >= label]
            else:
                curr_feats = features[label_bin == label]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[int(label - self.bucket_start)] = \
                (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
            self.running_var[int(label - self.bucket_start)] = \
                (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]

        logger.info(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features
        
        labels_0= labels[:, 0]
        label_bin= torch.Tensor([int((value-self.min_value)//self.bin_width) for value in labels_0])

        # labels = labels.squeeze(1)

        # labels = labels.reshape(labels.shape[0],-1)
        # labels = torch.mean(labels,dim=1).unsqueeze(-1)

        for label in torch.unique(label_bin):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                features[label_bin <= label] = calibrate_mean_var(
                    features[label_bin <= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            elif label == self.bucket_num - 1:
                features[label_bin >= label] = calibrate_mean_var(
                    features[label_bin >= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            else:
                features[label_bin == label] = calibrate_mean_var(
                    features[label_bin == label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
        # logger.info(f"Smoothed feature with Epoch [{epoch}]!")
        return features
