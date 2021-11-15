import gc
import logging
import sys
import os
import time
import torch
from tensorboardX import SummaryWriter

from data.datasets import get_dataloader
from data.utils.split_train_val_test import save_csv, load_csv
from ml.metrics.metrics import Metrics
from ml.models import get_model
from ml.modules.losses import get_loss
from ml.optimizers import get_optimizer, get_lr_policy, get_lr_policy_parameter
from tqdm import tqdm

from utils.device import DEVICE, put_minibatch_to_device
from utils.iohandler import IOHandler


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
        self.init_dataloaders()
        self.init_model()
        self.init_loss()
        self.init_optimizer()
        self.init_lr_policy()
        self.iohandler = IOHandler(args, self)

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
            self.val_loader = get_dataloader(self.config.data, 'val')
        if self.phase == 'val' or self.phase == 'test':
            self.val_loader = get_dataloader(self.config.data, self.phase)

    def run(self):
        logging.info("Starting experiment.")
        if self.phase == 'train':
            self.train()
        elif self.phase == 'val' or self.phase == 'test':
            self.eval()
        else:
            raise ValueError(f'Wrong phase: {self.phase}')

        return self.iohandler.get_max_metric()

    def before_epoch(self):
        """
        Before every epoch set the model to the right mode (train or eval)
        and select the corresponding loader and metric.
        """
        self.iohandler.reset_results()
        if self.current_mode == 'train':
            self.model.train()
            self.loader = self.train_loader
            self.loader.dataset.sample_classes()
            self.iohandler.train()
        elif self.current_mode == 'val':
            self.model.eval()
            self.loader = self.val_loader
            self.iohandler.val()
        else:
            raise ValueError(f'Wrong solver mode: {self.current_mode}')
        torch.cuda.empty_cache()

    def after_epoch(self):
        """
        After every epoch collect some garbage, evaluate and reset the current metric.
        """
        self.iohandler.compute_epoch_metric()
        gc.collect()
        torch.cuda.empty_cache()
        print()

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
            self.iohandler.save_best_checkpoint()
        self.iohandler.writer.close()

    def eval(self):
        self.current_mode = 'val'
        with torch.no_grad():
            self.run_epoch()
            self.iohandler.append_results_csv()

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
                minibatch = put_minibatch_to_device(minibatch)
                preproc_time = time.time() - preproc_t_start

                # train
                train_t_start = time.time()
                output, loss = self.step(minibatch)
                train_time = time.time() - train_t_start

                self.iohandler.append_results(minibatch, output)
                self.iohandler.calculate_iteration_metrics(minibatch, output, loss, pbar, preproc_time, train_time, idx)

                pbar.update(1)
                preproc_t_start = time.time()

        self.after_epoch()

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
