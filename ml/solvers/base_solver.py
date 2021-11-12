import gc
import logging
import sys
import os
import time
import torch

from data.datasets import get_dataloader
from ml.metrics.metrics import Metrics
from ml.models import get_model
from ml.modules.losses import get_loss
from ml.optimizers import get_optimizer, get_lr_policy, get_lr_policy_parameter
from tqdm import tqdm

from utils.device import DEVICE


class Solver(object):

    def __init__(self, config, args):
        """
        Solver parent function to control the experiments.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        self.args = args
        self.phase = args.mode
        self.config = config

        # initialize the required elements for the ml problem
        self.init_epochs()
        self.init_results_dir()
        self.init_dataloaders()
        self.init_model()
        self.init_loss()
        self.init_optimizer()
        self.init_lr_policy()
        self.init_metrics()

        self.load_checkpoint()

        # self.visualizer = Visualizer(self.writer)

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

    def init_epochs(self):
        """
        This function should implement the epoch number initialization(s).
        """
        logging.info("Initializing epoch number.")
        self.epoch = 0  # for resuming a training this should be changed
        self.epochs = self.config.env.epochs

    def init_model(self):
        """
        Initialize the model according to the config and put it on the gpu.
        """
        logging.info("Initializing the model.")
        self.model = get_model(self.config.model)
        if DEVICE == torch.device('cuda'): self.model.cuda()

    def init_loss(self):
        """
        Initialize the loss according to the config.
        """
        if self.phase == 'train':
            logging.info("Initializing the loss.")
            self.loss = get_loss(self.config.loss)

    def init_optimizer(self):
        """
        Initialize the optimizer according to the config.
        """
        if self.phase == 'train':
            logging.info("Initializing the optimizer.")
            self.optimizer = get_optimizer(self.config.optimizer, self.model.parameters())

    def init_lr_policy(self):
        """
        Initialize the learning rate policy.
        """
        if self.phase == 'train':
            logging.info("Initializing lr policy.")
            self.lr_policy = get_lr_policy(self.config.lr_policy, optimizer=self.optimizer)

    def init_dataloaders(self):
        """
        Dataloader initialization(s) for train, val and test dataset according to the config.
        """
        logging.info("Initializing dataloaders.")
        if self.phase == 'train':
            self.train_loader = get_dataloader(self.config.data, 'train')
        else:
            raise ValueError(f'Wrong args mode: {self.args.mode}')
        self.val_loader = get_dataloader(self.config.data, 'val')

    def init_metrics(self):
        """
        Initialize the metrics to follow the performance during training and validation (or testing).
        """
        logging.info("Initializing metrics.")
        if self.phase == 'train':
            self.train_metric = Metrics(self.result_dir, 'train', self.config.metrics)
        elif self.phase == 'test':
            return
        else:
            raise ValueError(f'Wrong args mode: {self.args.mode}')
        self.val_metric = Metrics(self.result_dir, 'val', self.config.metrics)

    def before_epoch(self):
        """
        Before every epoch set the model to the right mode (train or eval)
        and select the corresponding loader and metric.
        """
        self.results = {'paths': [], 'preds': [], 'labels': []}
        if self.current_mode == 'train':
            self.model.train()
            self.loader = self.train_loader
            self.metric = self.train_metric
        elif self.current_mode == 'val':
            self.model.eval()
            self.loader = self.val_loader
            self.metric = self.val_metric
        else:
            raise ValueError(f'Wrong solver mode: {self.current_mode}')
        torch.cuda.empty_cache()

    def after_epoch(self):
        """
        After every epoch collect some garbage, evaluate and reset the current metric.
        """
        if self.phase != 'test':
            self.metric.compute_epoch_end_metric(torch.cat(self.results['preds'], 0),
                                                 torch.cat(self.results['labels'], 0),
                                                 self.epoch)
        gc.collect()
        torch.cuda.empty_cache()
        print()

    def run(self):
        if self.phase == 'train':
            self.current_mode = 'train'
            self.train()
        elif self.phase == 'val' or self.phase == 'test':
            self.current_mode = 'val'
            self.eval()
        else:
            raise ValueError(f'Wrong phase: {self.phase}')

        return max(self.val_metric.epoch_results[self.config.metrics.goal_metric])

    def train(self):
        """
        Training all the epochs with validation after every epoch.
        Save the model if it has better performance than the previous ones.
        """

        for self.epoch in range(self.epoch, self.epochs):
            logging.info(f"Start training epoch: {self.epoch}/{self.epochs}")
            self.current_mode = 'train'
            self.run_epoch()
            logging.info(f"Start evaluating epoch: {self.epoch}/{self.epochs}")
            self.eval()
            self.lr_policy.step(*get_lr_policy_parameter(self))
            self.save_best_checkpoint()
        # self.writer.close()

    def eval(self):
        self.current_mode = 'val'
        with torch.no_grad():
            self.run_epoch()
            if self.phase != 'test': ...  # save results to csv

    def run_epoch(self):
        """
        Run a full epoch according to the current self.current_mode (train or val).
        """
        self.before_epoch()

        # set loading bar
        bar_format = '{desc}|{bar:10}|[{elapsed}<{remaining},{rate_fmt}]'
        with tqdm(range(len(self.loader)), file=sys.stdout, bar_format=bar_format, position=0, leave=True) as pbar:
            # start measuring preproc time
            preproc_t_start = time.time()
            for idx, minibatch in enumerate(self.loader):
                preproc_time = time.time() - preproc_t_start

                # train
                train_t_start = time.time()
                output, loss = self.step(minibatch)
                train_time = time.time() - train_t_start

                if self.phase != 'test':
                    self.metric.compute_metric(output, minibatch['labels'])

                    # save for calculating the full epoch metrics (the dataset is small so it can fit into the memory)
                    # the calculated metrics are `macro` average to be less sensitive to the class imbalance
                    # with larger datasets running metrics are suggested
                    # large datasets usually less affected by class imbalance too
                    self.results['paths'].append(minibatch['paths'])
                    self.results['preds'].append(output)
                    self.results['labels'].append(minibatch['labels'])

                    self.update_bar_description(pbar, idx, preproc_time, train_time, loss)
                    # self.write_to_tensorboard()

                pbar.update(1)
                preproc_t_start = time.time()

        self.after_epoch()

    def update_bar_description(self, pbar, idx, preproc_time, train_time, loss):
        """
        Update the current log bar with the latest result.

        :param pbar: pbar object
        :param idx: iteration number in the epoch
        :param preproc_time: time spent with preprocessing
        :param train_time: time spent with training
        :param loss: loss value
        """
        print_str = f'[{self.current_mode}] epoch {self.epoch}/{self.epochs} ' \
                    + f'iter {idx + 1}/{len(self.loader)}:' \
                    + f'lr:{self.optimizer.param_groups[0]["lr"]:.5f}|' \
                    + f'loss: {loss:.3f}|' \
                    + self.metric.get_snapshot_info() \
                    + f'|t_prep: {preproc_time:.3f}s|' \
                    + f't_train: {train_time:.3f}s'
        # + self.metric.get_snapshot_info() \
        pbar.set_description(print_str, refresh=False)

    def write_to_tensorboard(self):
        ...
        # write on tensorboard
        # if idx % self.config.env.save_train_frequency == 0:
        #     self.visualizer.visualize(minibatch, pred, self.epoch, tag='train')
        #     metric.add_scalar(self.writer, iteration=idx)

        # start measuring preproc time

    def step(self, minibatch):
        """
        Make one iteration step: either a train or a val step

        :param minibatch: minibatch containing the input image and the one-hot labels
        :return: dictionary of prediction and loss
        """
        output = self.model(minibatch['input_images'])

        if self.current_mode == 'train':
            # training step
            self.optimizer.zero_grad()
            loss = self.loss(output, minibatch['labels'])
            loss.backward()
            self.optimizer.step()
        else:
            loss = 0

        return output, loss

    def save_best_checkpoint(self):
        """
        Save the model if the last epoch result is the best.
        """
        epoch_results = self.val_metric.epoch_results[self.config.metrics.goal_metric]

        if not max(epoch_results) == epoch_results[-1]:
            return

        path = os.path.join(self.result_dir, 'model_best.pth.tar')

        state_dict = {'epoch': self.epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'lr_policy': self.lr_policy.state_dict(),
                      'train_metric': self.train_metric,
                      'val_metric': self.val_metric,
                      'config': self.config,
                      'model': self.model.state_dict()}

        torch.save(state_dict, path)
        del state_dict
        logging.info(f"Saved checkpoint to file {path}\n")

    def load_checkpoint(self):
        """
        If a saved model in the result folder exists load the model and the hyperparameters from a trained model checkpoint.
        """
        path = os.path.join(self.result_dir, 'model_best.pth.tar')
        if not os.path.exists(path):
            return

        logging.info(f"Loading the checkpoint from: {path}")
        continue_state_object = torch.load(path, map_location=torch.device("cpu"))

        # load the needed things from the checkpoint
        if self.phase == 'train':
            self.optimizer.load_state_dict(continue_state_object['optimizer'])
            self.lr_policy.load_state_dict(continue_state_object['lr_policy'])
            self.train_metric = continue_state_object['train_metric']
        if self.phase != 'test':
            self.val_metric = continue_state_object['val_metric']

        self.epoch = continue_state_object['epoch']
        self.config = continue_state_object['config']
        self.model.load_state_dict(continue_state_object['model'])
        if DEVICE == torch.device('cuda'): self.model.cuda()

        del continue_state_object
