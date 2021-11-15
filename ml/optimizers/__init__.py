from pytorch_ranger import Ranger
from torch_optimizer import RAdam
from torch import optim
import torch.optim.lr_scheduler as torch_lr_schedulers


def get_optimizer(optimizer_config, model_params):
    """
    Get the optimizer function according to the optimizer config name and parameters.

    :param optimizer_config: config containing the optimizer name as config.name and the parameters as config.params
    :param model_params: model parameters for the optimizer
    :return: the optimizer
    """
    if hasattr(optim, optimizer_config.name):
        function = getattr(optim, optimizer_config.name)
        return function(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'RAdam':
        return RAdam(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'Ranger':
        return Ranger(params=model_params, **optimizer_config.params.dict())
    else:
        raise ValueError(f'Wrong optimizer name: {optimizer_config.name}')


def get_lr_policy(lr_policy_config, optimizer):
    """
    Get the lr policy function according to the lr_policy config name and parameters.

    :param lr_policy_config: config containing the lr_policy name as config.name and the parameters as config.params
    :param optimizer: optimizer for the lr policy
    :return: the lr_policy
    """

    if hasattr(torch_lr_schedulers, lr_policy_config.name):
        function = getattr(torch_lr_schedulers, lr_policy_config.name)
        return function(optimizer=optimizer, **lr_policy_config.params.dict())
    else:
        raise ValueError(f'Wrong lr_policy name: {lr_policy_config.name}')


def get_lr_policy_parameter(solver):
    """
    Get the parameters for the lr policy function.
    Some lr policies need the epoch results (`like ReduceOnPlateau`), some need the current epoch number.

    :param solver: the solver object to reach the lr_policy and usable parameters
    :return: the lr_policy parameters in a list
    """

    if solver.lr_policy.__class__.__name__ == 'ReduceLROnPlateau':
        return [solver.iohandler.val_metric.epoch_results[solver.config.metrics.goal_metric][-1]]
    else:
        return [solver.epoch]
