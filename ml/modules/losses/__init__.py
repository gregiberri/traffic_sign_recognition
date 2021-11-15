# -*- coding: utf-8 -*-
# @Time    : 2021/11/11
# @Author  : Albert Gregus
# @Email   : g.albert95@gmail.com

import torch.nn as torch_losses
from ml.modules.losses.focal_loss import FocalLoss


def get_loss(loss_config):
    """
    Get the loss function according to the loss config name and parameters.

    :param loss_config: config containing the loss name as config.name and the parameters as config.params
    :return: the loss function
    """
    if hasattr(torch_losses, str(loss_config.name)):
        function = getattr(torch_losses, loss_config.name)
        return function(**loss_config.params.dict())
    if loss_config.name == 'FocalLoss':
        return FocalLoss(**loss_config.params.dict())
    else:
        raise ValueError(f'Wrong loss name: {loss_config.name}')
