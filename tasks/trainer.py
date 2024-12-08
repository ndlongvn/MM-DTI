# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from ast import Load

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_linear_schedule_with_warmup
from utils import Metrics
from utils import logger
from tasks.split import Splitter
from tqdm import tqdm

import time
import sys


class Trainer(object):
    """A :class:`Trainer` class is responsible for initializing the model, and managing its training, validation, and testing phases."""
    def __init__(self, save_path=None, **params):
        """
        :param save_path: Path for saving the training outputs. Defaults to None.
        :param params: Additional parameters for training.
        """
        self.save_path = save_path
        self.task = params.get('task', None)

        if self.task != 'repr':
            self.metrics_str = params['metrics']
            self.metrics = Metrics(self.task, self.metrics_str)
        self._init_trainer(**params)

    def _init_trainer(self, **params):
        """
        Initializing the trainer class to train model.

        :param params: Containing training arguments.
        """
        ### init common params ###
        self.seed = params.get('seed', 42)
        logger.info("Using seed: {}".format(self.seed))
        self.set_seed(self.seed)
        self.logger_level = int(params.get('logger_level', 1))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('learning_rate', 1e-4))
        self.batch_size = params.get('batch_size', 32)
        self.max_epochs = params.get('epochs', 50)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.max_norm = params.get('max_norm', 1.0)
        self.cuda = params.get('use_cuda', False)
        self.amp = params.get('use_amp', False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(
        ) if self.device.type == 'cuda' and self.amp == True else None
        self.alpha = params.get('alpha', 1)
        self.beta = params.get('beta', 0.1)
        self.fds= params.get('fds', False)

    def decorate_batch(self, batch, feature_name=None):
        """
        Prepares a batch of data for processing by the model. This method is a wrapper that
        delegates to a specific batch decoration method based on the data type.
        
        :param batch: The batch of data to be processed.
        :param feature_name: (str, optional) Name of the feature used in batch decoration. Defaults to None.

        :return: The decorated batch ready for processing by the model.
        """
        return self.decorate_torch_batch(batch)

    def decorate_graph_batch(self, batch):
        """
        Prepares a graph-based batch of data for processing by the model. Specifically handles 
        graph-based data structures.
        
        :param batch: The batch of graph-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = {'net_input': batch.to(
            self.device)}, batch.y.to(self.device)
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def decorate_torch_batch(self, batch):
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        # print(net_input)
        if isinstance(net_input, dict):
            net_input, net_target = {
                k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input': net_input.to(
                self.device)}, net_target.to(self.device)
        if self.task == 'repr':
            net_target = None
        elif self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        # print(net_input)
        return net_input, net_target

    def fit_predict(self, model, train_dataset, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, feature_name=None, return_infonce_loss= False, return_ct_loss= False, use_weight= False):
        """
        Trains the model on the given training dataset and evaluates it on the validation dataset.

        :param model: The model to be trained and evaluated.
        :param train_dataset: Dataset used for training the model.
        :param valid_dataset: Dataset used for validating the model.
        :param loss_func: The loss function used during training.
        :param activation_fn: The activation function applied to the model's output.
        :param dump_dir: Directory where the best model state is saved.
        :param fold: The fold number in a cross-validation setting.
        :param target_scaler: Scaler used for scaling the target variable.
        :param feature_name: (optional) Name of the feature used in data loading. Defaults to None.
        
        :return: Predictions made by the model on the validation dataset.
        """
        model = model.to(self.device)
        train_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=model.batch_collate_fn,
            drop_last=True,
        )
        
        
        # remove last batch, bs=1 can not work on batchnorm1d
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        ### init optimizer ###
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, eps=1e-6)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        for epoch in range(self.max_epochs):
            model = model.train()
            # Progress Bar
            start_time = time.time()
            batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True,
                             leave=False, position=0, desc='Train', ncols=5)
            trn_loss = []
            if return_infonce_loss:
                trn_infonce_loss = []
            if return_ct_loss:
                trn_ct_loss = []
            trn_m_loss= []

            for i, batch in enumerate(train_dataloader):
                net_input, net_target = self.decorate_batch(
                    batch, feature_name)
                optimizer.zero_grad()  # Zero gradients
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast( dtype=torch.float16):
                        if not return_infonce_loss and not return_ct_loss:
                            outputs = model(**net_input, epoch= epoch)
                            loss = self.alpha*loss_func(outputs, net_target)
                            trn_m_loss.append(float(loss_func(outputs, net_target).data))
                        else:
                            if return_ct_loss and return_infonce_loss:
 
                                outputs, ct_loss, rnc_loss = model(**net_input, return_infonce_loss= return_infonce_loss, return_ct_loss= return_ct_loss, net_target= net_target, use_weight= use_weight, epoch= epoch)

                                rg_loss= loss_func(outputs, net_target)
                                loss = self.alpha*rg_loss + self.beta*ct_loss + self.beta*rnc_loss
                                # add data to list
                                trn_infonce_loss.append(float(ct_loss.data))
                                trn_ct_loss.append(float(rnc_loss.data))
                                trn_m_loss.append(float(rg_loss.data))
                            elif not return_ct_loss and return_infonce_loss:
                                outputs, ct_loss = model(**net_input, return_infonce_loss= return_infonce_loss, epoch= epoch, net_target= net_target, )
                                loss = self.alpha*loss_func(outputs, net_target) + self.beta*ct_loss
                                # add data to list
                                trn_infonce_loss.append(float(ct_loss.data))
                                trn_m_loss.append(float(loss_func(outputs, net_target).data))
                            elif return_ct_loss and not return_infonce_loss:
                                outputs, rnc_loss = model(**net_input, return_ct_loss= return_ct_loss, net_target= net_target, use_weight= use_weight, epoch= epoch)
                                loss = self.alpha*loss_func(outputs, net_target) + self.beta*rnc_loss
                                # add data to list
                                trn_ct_loss.append(float(rnc_loss.data))
                                trn_m_loss.append(float(loss_func(outputs, net_target).data))

                else:
                    with torch.set_grad_enabled(True):
                        if not return_infonce_loss and not return_ct_loss:
                            outputs = model(**net_input, epoch= epoch)
                            loss = self.alpha*loss_func(outputs, net_target)
                            trn_m_loss.append(float(loss_func(outputs, net_target).data))
                        else:
                            if return_ct_loss and return_infonce_loss:
                                outputs, ct_loss, rnc_loss = model(**net_input, return_infonce_loss= return_infonce_loss, return_ct_loss= return_ct_loss, net_target= net_target, use_weight= use_weight, epoch= epoch)
  
                                loss = self.alpha*loss_func(outputs, net_target) + self.beta*ct_loss + self.beta*rnc_loss
                                # add data to list
                                trn_infonce_loss.append(float(ct_loss.data))
                                trn_ct_loss.append(float(rnc_loss.data))
                                trn_m_loss.append(float(loss_func(outputs, net_target).data))
                            elif not return_ct_loss and return_infonce_loss:
                                outputs, ct_loss = model(**net_input, return_infonce_loss= return_infonce_loss, epoch= epoch)
                                loss = self.alpha*loss_func(outputs, net_target) + self.beta*ct_loss
                                # add data to list
                                trn_infonce_loss.append(float(ct_loss.data))
                                trn_m_loss.append(float(loss_func(outputs, net_target).data))
                            elif return_ct_loss and not return_infonce_loss:
                                outputs, rnc_loss = model(**net_input, return_ct_loss= return_ct_loss, net_target= net_target, use_weight= use_weight, epoch= epoch)
                                loss = self.alpha*loss_func(outputs, net_target) + self.beta*rnc_loss
                                # add data to list
                                trn_ct_loss.append(float(rnc_loss.data))
                                trn_m_loss.append(float(loss_func(outputs, net_target).data))
                trn_loss.append(float(loss.data))
                # tqdm lets you add some details so you can monitor training as you train.
                if not return_infonce_loss and not return_ct_loss:
                    batch_bar.set_postfix(
                        Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                        loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                        m_loss="{:.04f}".format(float(sum(trn_m_loss) / (i + 1))),
                        lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                elif return_infonce_loss and return_ct_loss:
                    batch_bar.set_postfix(
                        Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                        loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                        m_loss="{:.04f}".format(float(sum(trn_m_loss) / (i + 1))),
                        infonce_loss="{:.04f}".format(float(sum(trn_infonce_loss) / (i + 1))),
                        ct_loss="{:.04f}".format(float(sum(trn_ct_loss) / (i + 1))),
                        lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                elif return_infonce_loss and not return_ct_loss:
                    batch_bar.set_postfix(
                        Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                        loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                        m_loss="{:.04f}".format(float(sum(trn_m_loss) / (i + 1))),
                        infonce_loss="{:.04f}".format(float(sum(trn_infonce_loss) / (i + 1))),
                        lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                elif return_ct_loss and not return_infonce_loss:
                    batch_bar.set_postfix(
                        Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                        loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                        m_loss="{:.04f}".format(float(sum(trn_m_loss) / (i + 1))),
                        ct_loss="{:.04f}".format(float(sum(trn_ct_loss) / (i + 1))),
                        lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                if self.scaler and self.device.type == 'cuda':
                    # This is a replacement for loss.backward()
                    self.scaler.scale(loss).backward()
                    # unscale the gradients of optimizer's assigned params in-place
                    self.scaler.unscale_(optimizer)
                    # Clip the norm of the gradients to max_norm.
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    # This is a replacement for optimizer.step()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # clip_grad_norm_(model.parameters(), self.max_norm)
                    optimizer.step()
                scheduler.step()
                batch_bar.update()

            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)

            if self.fds and epoch >= model.fds_cfg.start_update:
                logger.info(f"Create Epoch [{epoch}] features of all training data...")
                encodings, labels = [], []
                with torch.no_grad():
                    batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True,
                             leave=False, position=0,  desc="FDS Epoch {}".format(epoch), ncols=5)
                    for i, batch in enumerate(train_dataloader):
                        net_input, net_target = self.decorate_batch(
                            batch, feature_name)
                        outputs, feature = model(**net_input, epoch=epoch, return_feature= True, net_target= net_target)
                        encodings.append(feature.squeeze().cpu().numpy())
                        labels.append(net_target.squeeze().cpu().numpy())
                        batch_bar.update()
                    batch_bar.close()
  
                encodings = torch.from_numpy(np.concatenate(encodings)).cuda()
                labels = torch.from_numpy(np.concatenate(labels)).cuda()
                model.FDS.update_last_epoch_stats(epoch)
                model.FDS.update_running_stats(encodings, labels, epoch)

            y_preds, val_loss, metric_score = self.predict(
                model, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, epoch, load_model=False, feature_name=feature_name, return_infonce_loss= False)
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = list(metric_score.values())[0]
            _metric = list(metric_score.keys())[0]
            if return_infonce_loss and return_ct_loss:
                message= 'Epoch [{}/{}] train_loss: {:.4f}, train_m_loss: {:.4f}, train_infonce_loss: {:.4f}, train_ct_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch+1, self.max_epochs, total_trn_loss, np.mean(trn_m_loss), np.mean(trn_infonce_loss), np.mean(trn_ct_loss), total_val_loss, _metric, _score, optimizer.param_groups[0]['lr'], (end_time - start_time))
            elif not return_infonce_loss and return_ct_loss:
                message= 'Epoch [{}/{}] train_loss: {:.4f}, train_rg_loss: {:.4f}, train_rnc_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch+1, self.max_epochs, total_trn_loss, np.mean(trn_m_loss), np.mean(trn_ct_loss), total_val_loss, _metric, _score, optimizer.param_groups[0]['lr'], (end_time - start_time))
            else:
                message= 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch+1, self.max_epochs, total_trn_loss, total_val_loss, _metric, _score, optimizer.param_groups[0]['lr'], (end_time - start_time))
            # logger.info(message)
            logger.info(message)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(
                wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, fold, self.patience, epoch)
            if is_early_stop:
                break

        y_preds, _, _ = self.predict(model, valid_dataset, loss_func, activation_fn,
                                     dump_dir, fold, target_scaler, epoch, load_model=True, feature_name=feature_name)
        return y_preds

    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        """
        Determines if early stopping criteria are met, based on either loss improvement or custom metric score.

        :param wait: Number of epochs waited since the last improvement in loss or metric score.
        :param loss: The current loss value.
        :param min_loss: The minimum loss value observed so far.
        :param metric_score: Current metric score.
        :param max_score: The maximum metric score observed so far.
        :param model: The model being trained.
        :param dump_dir: Directory to save the best model state.
        :param fold: The fold number in cross-validation.
        :param patience: Number of epochs to wait for an improvement before stopping.
        :param epoch: The current epoch number.

        :return: A tuple (is_early_stop, min_val_loss, wait, max_score) indicating if early stopping criteria are met, the minimum validation loss, the updated wait time, and the maximum metric score.
        """
        if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
            is_early_stop, min_val_loss, wait = self._judge_early_stop_loss(
                wait, loss, min_loss, model, dump_dir, fold, patience, epoch)
        else:
            is_early_stop, min_val_loss, wait, max_score = self.metrics._early_stop_choice(
                wait, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop, min_val_loss, wait, max_score

    def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, fold, patience, epoch):
        """
        Determines whether early stopping should be triggered based on the loss comparison.

        :param wait: The number of epochs to wait after min_loss has stopped improving.
        :param loss: The current loss value of the model.
        :param min_loss: The minimum loss value observed so far.
        :param model: The neural network model being trained.
        :param dump_dir: Directory to save the model state.
        :param fold: The current fold number in a cross-validation setting.
        :param patience: The number of epochs to wait for an improvement before stopping.
        :param epoch: The current epoch number.

        :return: A tuple (is_early_stop, min_loss, wait), where is_early_stop is a boolean indicating 
                 whether early stopping should occur, min_loss is the updated minimum loss, 
                 and wait is the updated wait counter.
        """
        is_early_stop = False
        if loss <= min_loss:
            min_loss = loss
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait

    def predict(self, model, dataset, loss_func, activation_fn, dump_dir, fold, target_scaler=None, epoch=1, load_model=False, feature_name=None, return_infonce_loss= False, return_ct_loss= False, return_feature= False):
        """
        Executes the prediction on a given dataset using the specified model.

        :param model: The model to be used for predictions.
        :param dataset: The dataset to perform predictions on.
        :param loss_func: The loss function used during training.
        :param activation_fn: The activation function applied to the model's output.
        :param dump_dir: Directory where the model state is saved.
        :param fold: The fold number in cross-validation.
        :param target_scaler: (optional) Scaler to inverse transform the model's output. Defaults to None.
        :param epoch: (int) The current epoch number. Defaults to 1.
        :param load_model: (bool) Whether to load the model from a saved state. Defaults to False.
        :param feature_name: (str, optional) Name of the feature for data processing. Defaults to None.

        :return: A tuple (y_preds, val_loss, metric_score), where y_preds are the predicted outputs, 
                 val_loss is the validation loss, and metric_score is the calculated metric score.
        """
        model = model.to(self.device)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model_{fold}.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)[
                "model_state_dict"]
            model.load_state_dict(model_dict)
            logger.info("load model success!")
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='Val', ncols=5)
        val_loss = []
        y_preds = []
        y_truths = []

        # disable 
        return_infonce_loss= False
        return_ct_loss= False
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch, feature_name)
            # Get model outputs
            with torch.no_grad():
                if not return_infonce_loss and not return_ct_loss:
                    outputs = model(**net_input)
                    if not load_model:
                        loss = self.alpha*loss_func(outputs, net_target)
                        val_loss.append(float(loss.data))
                else:
                    if return_ct_loss and return_infonce_loss:
                        outputs, ct_loss, rnc_loss = model(**net_input, return_infonce_loss= return_infonce_loss, return_ct_loss= return_ct_loss, net_target= net_target)
                        loss = self.alpha*loss_func(outputs, net_target) + self.beta*ct_loss + self.beta*rnc_loss
                        # val_loss.append(float(loss.data))
                    elif not return_ct_loss and return_infonce_loss:
                        outputs, ct_loss = model(**net_input, return_infonce_loss= return_infonce_loss)
                        loss = self.alpha*loss_func(outputs, net_target) + self.beta*ct_loss
                        # val_loss.append(float(loss.data))
                    elif return_ct_loss and not return_infonce_loss:
                        outputs, rnc_loss = model(**net_input, return_ct_loss= return_ct_loss, net_target= net_target)
                        loss = self.alpha*loss_func(outputs, net_target) + self.beta*rnc_loss
    
                if not load_model:
    
                    val_loss.append(float(loss.data))
            # print(activation_fn(outputs).detach().cpu().numpy())
            y_preds.append(activation_fn(outputs).detach().cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))))

            batch_bar.update()
        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)

        try:
            label_cnt = model.output_dim
        except:
            label_cnt = None
        if self.alpha!=0:
            if target_scaler is not None:
                inverse_y_preds = target_scaler.inverse_transform(y_preds)
                inverse_y_truths = target_scaler.inverse_transform(y_truths)
                metric_score = self.metrics.cal_metric(
                    inverse_y_truths, inverse_y_preds, label_cnt=label_cnt) if not load_model else None
            else:
                metric_score = self.metrics.cal_metric(
                    y_truths, y_preds, label_cnt=label_cnt) if not load_model else None
        else:
            metric_score= {"ct_loss":np.mean(val_loss)}
        batch_bar.close()
        return y_preds, val_loss, metric_score

    def inference(self, model, dataset, return_repr=False, return_atomic_reprs=False, feature_name=None):
        """
        Runs inference on the given dataset using the provided model. This method can return
        various representations based on the model's output.

        :param model: The neural network model to be used for inference.
        :param dataset: The dataset on which inference is to be performed.
        :param return_repr: (bool, optional) If True, returns class-level representations. Defaults to False.
        :param return_atomic_reprs: (bool, optional) If True, returns atomic-level representations. Defaults to False.
        :param feature_name: (str, optional) Name of the feature used for data loading. Defaults to None.

        :return: A dictionary containing different types of representations based on the model's output and the
                 specified parameters. This can include class-level representations, atomic coordinates, 
                 atomic representations, and atomic symbols.
        """
        model = model.to(self.device)
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        repr_dict = {"cls_repr": [], "atomic_coords": [], "atomic_reprs": [], "atomic_symbol": []}
        for batch in tqdm(dataloader):
            net_input, _ = self.decorate_batch(batch, feature_name)
            with torch.no_grad():
                outputs = model(**net_input,
                                return_repr=return_repr,
                                return_atomic_reprs=return_atomic_reprs)
                assert isinstance(outputs, dict)
                repr_dict["cls_repr"].extend(item.cpu().numpy() for item in outputs["cls_repr"])
                if return_atomic_reprs:
                    repr_dict["atomic_symbol"].extend(outputs["atomic_symbol"])
                    repr_dict['atomic_coords'].extend(item.cpu().numpy() for item in outputs['atomic_coords'])
                    repr_dict['atomic_reprs'].extend(item.cpu().numpy() for item in outputs['atomic_reprs'])
                    
        return repr_dict

    def set_seed(self, seed):
        """
        Sets a random seed for torch and numpy to ensure reproducibility.
        :param seed: (int) The seed number to be set.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def NNDataLoader(feature_name=None, dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False):
    """
    Creates a DataLoader for neural network training or inference. This function is a wrapper 
    around the standard PyTorch DataLoader, allowing for custom feature handling and additional 
    configuration.

    :param feature_name: (str, optional) Name of the feature used for data loading. 
                         This can be used to specify a particular type of data processing. Defaults to None.
    :param dataset: (Dataset, optional) The dataset from which to load the data. Defaults to None.
    :param batch_size: (int, optional) Number of samples per batch to load. Defaults to None.
    :param shuffle: (bool, optional) Whether to shuffle the data at every epoch. Defaults to False.
    :param collate_fn: (callable, optional) Merges a list of samples to form a mini-batch. Defaults to None.
    :param drop_last: (bool, optional) Set to True to drop the last incomplete batch. Defaults to False.

    :return: DataLoader configured according to the provided parameters.
    """
    dataloader = TorchDataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=drop_last)
    return dataloader
