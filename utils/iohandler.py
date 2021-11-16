import logging
import os

import torch
from tensorboardX import SummaryWriter

from data.utils.split_train_val import load_csv, save_csv
from ml.metrics.metrics import Metrics
from utils.device import DEVICE


class IOHandler:
    def __init__(self, args, solver):
        self.args = args
        self.phase = args.mode
        self.solver = solver
        self.config = self.solver.config

        self.init_results_dir()
        self.init_metrics()
        self.init_tensorboard()
        self.load_checkpoint()
        self.reset_results()

    def train(self):
        """
        Set the iohandler to use the train metric.
        """
        self.metric = self.train_metric

    def val(self):
        """
        Set the iohandler to use the train metric.
        """
        if self.phase != 'test': self.metric = self.val_metric

    def reset_results(self):
        """
        Reset the results to be empty before starting an epoch.
        """
        self.results = {'paths': [], 'preds': [], 'labels': []}

    def get_max_metric(self):
        """
        Get the validation goal_metrics of the best model.
        """
        return max(self.val_metric.epoch_results[self.config.metrics.goal_metric]) if self.phase == 'train' else None

    def init_results_dir(self):
        """
        Making results dir.
        """
        logging.info("Making result dir.")
        result_name = os.path.join(self.config.id, self.args.id_tag) if self.args.id_tag else self.config.id
        self.result_dir = os.path.join(self.config.env.result_dir, result_name)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        logging.info(f"Results dir is made. Results will be saved at: {self.result_dir}")

    def init_metrics(self):
        """
        Initialize the metrics to follow the performance during training and validation (or testing).
        """
        if self.phase != 'test':
            logging.info("Initializing metrics.")
            self.val_metric = Metrics(self.result_dir, 'val', self.config.metrics)
        if self.phase == 'train':
            self.train_metric = Metrics(self.result_dir, 'train', self.config.metrics)

    def init_tensorboard(self):
        """
        Initialize the tensorboard.
        """
        if self.phase != 'test':
            logging.info("Initializing lr policy.")
            self.writer = SummaryWriter(os.path.join(self.result_dir, 'tensorboard'))

    def save_results_csv(self):
        """
        Save the data paths and predictions for test inference into a csv file.
        """
        if self.phase == 'test':
            # save results to csv
            test_result_file = os.path.join(self.result_dir, 'test_results.csv')
            classnumber_classnames = dict(load_csv('classnumber_classname.csv'))
            pred_classnames = [classnumber_classnames[str(int(torch.argmax(pred).detach().cpu()))]
                               for pred in self.results['preds']]
            paths = [path for sublist in self.results['paths'] for path in sublist]
            save_csv([paths, pred_classnames], test_result_file)
            logging.info(f'Predictions are saved into file: {test_result_file}')

    def append_results(self, minibatch, output):
        """
        Save for calculating the full epoch metrics (the dataset is small so it can fit into the memory)
        the calculated metrics are `macro` average to be less sensitive to the class imbalance
        with larger datasets running metrics are suggested
        large datasets usually less affected by class imbalance too.

        :param minibatch: minibatch diractory containing the paths (and during train or val the labels)
        :param output: predictions of the network
        """

        self.results['paths'].append(minibatch['paths'])
        self.results['preds'].append(output)

        if self.phase != 'test':
            self.results['labels'].append(minibatch['labels'])

    def calculate_iteration_metrics(self, minibatch, output, loss, pbar, preproc_time, train_time, idx):
        """
        Calculate the metrics during an iteration inside an epoch.
        """
        if self.phase != 'test':
            self.metric.compute_metric(output, minibatch['labels'])
            self.update_bar_description(pbar, idx, preproc_time, train_time, loss)

            # write to tensorboard
            writer_iteration = self.solver.epoch * len(self.solver.loader) + idx
            # image denormalization is hardcoded here which is not nice but at least works :P
            self.writer.add_image(f'{self.solver.current_mode}/image', minibatch['input_images'][0].cpu() * \
                                  torch.Tensor([[[0.1560]], [[0.1815]], [[0.1727]]]) + \
                                  torch.Tensor([[[0.4432]], [[0.3938]], [[0.3764]]]),
                                  writer_iteration)
            if self.solver.current_mode == 'train':
                self.writer.add_scalar('loss', loss, writer_iteration)

    def update_bar_description(self, pbar, idx, preproc_time, train_time, loss):
        """
        Update the current log bar with the latest result.

        :param pbar: pbar object
        :param idx: iteration number in the epoch
        :param preproc_time: time spent with preprocessing
        :param train_time: time spent with training
        :param loss: loss value
        """
        print_str = f'[{self.solver.current_mode}] epoch {self.solver.epoch}/{self.solver.epochs} ' \
                    + f'iter {idx + 1}/{len(self.solver.loader)}:' \
                    + f'lr:{self.solver.optimizer.param_groups[0]["lr"]:.5f}|' \
                    + f'loss: {loss:.3f}|' \
                    + self.metric.get_snapshot_info() \
                    + f'|t_prep: {preproc_time:.3f}s|' \
                    + f't_train: {train_time:.3f}s'
        pbar.set_description(print_str, refresh=False)

    def compute_epoch_metric(self):
        """
        Calculate the metrics of a whole epoch.
        """
        if self.phase != 'test':
            self.metric.compute_epoch_metric(torch.cat(self.results['preds'], 0),
                                             torch.cat(self.results['labels'], 0),
                                             self.writer,
                                             self.solver.epoch)
            for key, value in self.metric.current_metric.items():
                self.writer.add_scalar(f'{self.solver.current_mode}/{key}', value, self.solver.epoch)

    def save_best_checkpoint(self):
        """
        Save the model if the last epoch result is the best.
        """
        epoch_results = self.val_metric.epoch_results[self.config.metrics.goal_metric]

        if not max(epoch_results) == epoch_results[-1]:
            return

        path = os.path.join(self.result_dir, 'model_best.pth.tar')

        state_dict = {'epoch': self.solver.epoch,
                      'optimizer': self.solver.optimizer.state_dict(),
                      'lr_policy': self.solver.lr_policy.state_dict(),
                      'train_metric': self.train_metric,
                      'val_metric': self.val_metric,
                      'config': self.solver.config,
                      'model': self.solver.model.state_dict()}

        torch.save(state_dict, path)
        del state_dict
        logging.info(f"Saved checkpoint to file {path}\n")

    def load_checkpoint(self):
        """
        If a saved model in the result folder exists load the model
        and the hyperparameters from a trained model checkpoint.
        """
        path = os.path.join(self.result_dir, 'model_best.pth.tar')
        if not os.path.exists(path):
            assert self.phase == 'train', f'No model file found to load at: {path}'
            return

        logging.info(f"Loading the checkpoint from: {path}")
        continue_state_object = torch.load(path, map_location=torch.device("cpu"))

        # load the things from the checkpoint
        if self.phase == 'train':
            self.solver.optimizer.load_state_dict(continue_state_object['optimizer'])
            self.solver.lr_policy.load_state_dict(continue_state_object['lr_policy'])
            self.train_metric = continue_state_object['train_metric']
        if self.phase != 'test':
            self.val_metric = continue_state_object['val_metric']

        self.solver.epoch = continue_state_object['epoch']
        self.solver.config = continue_state_object['config']
        self.solver.model.load_state_dict(continue_state_object['model'])
        if DEVICE == torch.device('cuda'): self.solver.model.cuda()

        del continue_state_object
