from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch

modules = [element for element in dir() if '__' not in element]


def get_hpo_algorithm(hpo_config):
    """
    Get the hpo algorithm according to the hpo config name and parameters.

    :param hpo_config: config containing the hpo name as config.name and the parameters as config.params
    :return: the hpo algorithm object
    """

    if hpo_config.name in modules:
        function = globals()[hpo_config.name]
        return function(**hpo_config.params.dict())
    else:
        raise ValueError(f'Wrong hyperoptimizer name: {hpo_config.name}')
