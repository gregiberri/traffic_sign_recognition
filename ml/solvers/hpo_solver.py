import os
import sys

from ray import tune

from config import ConfigNamespace
from ml.hpo import get_hpo_algorithm
from ml.solvers.base_solver import Solver


class HPOSolver(object):
    def __init__(self, config, args):
        """
        Solver parent function for hyperparameter optimization and control the experiments.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        self.args = args
        self.phase = args.mode
        self.config = config

        self.experiment_number = 0

        self.init_results_dir()

    def init_results_dir(self):
        self.result_dir = os.path.join(self.config.env.result_dir, self.config.id)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def run(self):
        """
        Run a ray tuning controlling the
        """
        config_dict = self.config.dict()
        ray_result_dir = os.path.join(self.result_dir, 'tune')
        search_alg = get_hpo_algorithm(self.config.hpo.hpo_algorithm)
        result = tune.run(self.run_experiment,
                          config=config_dict,
                          num_samples=self.config.hpo.num_samples,
                          checkpoint_at_end=True,
                          queue_trials=True, local_dir=ray_result_dir,
                          resources_per_trial={"cpu": self.config.hpo.cpu_per_trial,
                                               "gpu": self.config.hpo.gpu_per_trial},
                          search_alg=search_alg)

        self.report_result(result)

    def report_result(self, result):
        """
        Write out the result to the command line and save it to a txt file.

        :param result: result file from the tune run
        """
        best_trial = result.get_best_trial('accuracy', 'max', 'last')
        print("Best trial config: {}".format(best_trial.logdir))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result['_metric']))

        result_file = os.path.join(self.result_dir, best_trial.last_result['date'] + '_result.txt')
        with open(result_file, 'w') as f:
            f.write("Best trial logdir: \n{}".format(best_trial.logdir))
            f.write("\n\nBest trial final validation accuracy: \n{}".format(best_trial.last_result['_metric']))

    def run_experiment(self, search_space):
        """
        Run an experiment with the given search space: make a Solver, run a training and get the best eval.

        :param search_space: config dict

        :return: the metric
        """
        # set current workdir to project dir (w/o it the datareader cant reach dataset_files)
        current_working_dir = __file__
        new_working_dir = '/' + os.path.join(*current_working_dir.split('/')[:-3])
        os.chdir(new_working_dir)
        sys.path.insert(0, new_working_dir)

        args = self.args
        args.mode = 'train'
        args.id_tag = os.path.join(*tune.get_trial_dir().split('/')[-3:-1])

        config = ConfigNamespace(search_space)
        config.id = os.path.join(config.id, 'outputs')

        solver = Solver(config, args)
        metric = solver.run()

        tune.report(accuracy=max(metric.epoch_results['acc']))

        return max(metric.epoch_results['acc'])
