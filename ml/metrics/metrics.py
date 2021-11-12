# -*- coding: utf-8 -*-
# @Time    : 2021/11/11
# @Author  : Albert Gregus
# @Email   : g.albert95@gmail.com

import csv
import logging
import os
import matplotlib
import torch

from ml.metrics import get_metric

matplotlib.use('Agg')


class Metrics(object):

    def __init__(self, save_dir='', tag='train', config=None):
        self.config = config
        self.metrics = {key: get_metric(key, value) for key, value in config.save_metrics.dict().items()}
        self.epoch_results = {key: [] for key in config.save_metrics.dict().keys()}
        self.result_names = ['epoch'] + list(self.metrics.keys())

        self.save_dir = save_dir
        self.tag = tag
        self.make_metrics_file()

        self.n_stage = -1

    def make_metrics_file(self):
        file_path = os.path.join(self.save_dir, f'{self.tag}_results.csv')
        if not os.path.exists(file_path):
            with open(file_path, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.result_names)
                writer.writeheader()

    def compute_metric(self, pred, gt):
        self.current_metric = {key: float(metric(pred.detach().cpu(), gt.detach().cpu()))
                               for key, metric in self.metrics.items()}
        return self.current_metric

    def get_snapshot_info(self):
        snapsnot_list = [f'{key[:4]}: {value:.3f}' for key, value in self.current_metric.items()]
        return '|'.join(snapsnot_list)

    def save_metrics(self, metric):
        # make result dict
        with open(os.path.join(self.save_dir, f'{self.tag}_results.csv'), mode='a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.result_names)
            writer.writerow(metric)

    def compute_epoch_end_metric(self, epoch_pred, epoch_gt, epoch):
        epoch_metric = self.compute_metric(epoch_pred, epoch_gt)
        self.save_metrics(epoch_metric)
        self.epoch_results = {key: results + [epoch_metric[key]] for key, results in self.epoch_results.items()}

        logging.info(f'The results of epoch {epoch} {self.tag} are: {self.get_snapshot_info()}')
