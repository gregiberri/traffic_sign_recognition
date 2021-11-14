# -*- coding: utf-8 -*-
# @Time    : 2021/11/11
# @Author  : Albert Gregus
# @Email   : g.albert95@gmail.com

import csv
import logging
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
import pandas as pd
from data.utils.split_train_val_test import load_csv
from ml.metrics import get_metric
import seaborn as sn

matplotlib.use('Agg')


class Metrics(object):

    def __init__(self, save_dir='', tag='train', config=None):
        self.config = config
        self.metrics = {key: get_metric(key, value) for key, value in config.save_metrics.dict().items()}
        self.epoch_results = {key: [] for key in config.save_metrics.dict().keys()}
        self.result_names = ['epoch'] + list(self.metrics.keys())
        self.num_classes = list(self.config.save_metrics.dict().values())[0]['num_classes']

        self.goal_metric_by_class = get_metric(self.config.goal_metric,
                                               {'num_classes': self.num_classes,
                                                'average': None})

        self.save_dir = save_dir
        self.by_class_result_dir = os.path.join(self.save_dir, 'by_class_results')
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

    def compute_epoch_end_metric(self, epoch_pred, epoch_gt, writer, epoch):
        epoch_metric = self.compute_metric(epoch_pred, epoch_gt)
        self.save_metrics(epoch_metric)
        self.epoch_results = {key: results + [epoch_metric[key]] for key, results in self.epoch_results.items()}
        self.draw_goal_metric_by_class(epoch_pred, epoch_gt, writer, epoch)
        logging.info(f'The results of epoch {epoch} {self.tag} are: {self.get_snapshot_info()}')

    def draw_goal_metric_by_class(self, epoch_pred, epoch_gt, writer, epoch):
        if not os.path.exists(self.by_class_result_dir):
            os.makedirs(self.by_class_result_dir)

        goal_metric = self.config.goal_metric

        fig = plt.figure(figsize=[15, 5])
        goal_metric_by_class = self.goal_metric_by_class(epoch_pred.detach().cpu(), epoch_gt.detach().cpu())
        goal_metric_by_class = [float(element) for element in goal_metric_by_class]
        number_class = dict(load_csv('classnumber_classname.csv'))

        # sort the classes according to the class size
        sorting_indices = np.argsort(list(number_class.values()))
        goal_metric_by_class = np.array(goal_metric_by_class)[sorting_indices]
        class_names = np.array(list(number_class.values()))[sorting_indices]

        plt.bar(class_names, goal_metric_by_class)
        plt.xticks(rotation=90)
        plt.title(f'{goal_metric} scores')
        plt.xlabel('Class')
        plt.ylabel(f'{goal_metric} score')
        for i in range(len(goal_metric_by_class)):
            plt.annotate(f'{goal_metric_by_class[i]:.3f}',
                         xy=(class_names[i], goal_metric_by_class[i]),
                         ha='center', va='bottom')
        plt.savefig(os.path.join(self.by_class_result_dir, f'{self.config.goal_metric}_{self.tag}_{epoch}.png'))
        writer.add_figure(f'{self.tag}/goal_metric_by_class', fig, epoch)
        plt.close(fig)

        # making confusion matrix
        conf_matrix = ConfusionMatrix(self.num_classes)
        array = conf_matrix(epoch_pred.detach().cpu(), epoch_gt.detach().cpu())
        df_cm = pd.DataFrame(array.numpy(), list(number_class.values()), list(number_class.values()))
        fig = plt.figure(figsize=(10, 7))
        sn.set(font_scale=0.7)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})  # font size
        plt.savefig(os.path.join(self.by_class_result_dir, f'confusion_matrix_{self.tag}_{epoch}.png'))
        writer.add_figure(f'{self.tag}/confusion_matrix', fig, epoch)
        plt.close(fig)
