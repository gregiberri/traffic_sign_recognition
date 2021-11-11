from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.skopt import SkOptSearch


def get_hpo_algorithm(config):
    """
    Choose which hyperparameter optimization algorithm to use.

    :param config: config containing the name and the parameters of the hpo algo

    :return: hpo algorithm
    """
    if config.name == 'default':
        return None
    elif config.name == 'bayesian':
        return BayesOptSearch(**config.params.dict())
    elif config.name == 'bohb':
        return TuneBOHB(**config.params.dict())
    elif config.name == 'dragonfly':
        return DragonflySearch(**config.params.dict())
    elif config.name == 'hebo':
        return HyperOptSearch(**config.params.dict())
    elif config.name == 'scikit':
        return SkOptSearch(**config.params.dict())
    else:
        raise ValueError(f'Wrong hyperoptimizer name: {config.name}')
