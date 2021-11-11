# -*- coding: utf-8 -*-
# @Time    : 2021/11/11
# @Author  : Albert Gregus
# @Email   : g.albert95@gmail.com
import torchmetrics


def get_metric(metric_name, params):
    """
    Get the metric function according to the config name and parameters.

    :param metric_name: config containing the metric name as config.name and the parameters as config.params
    :return: the metric function
    """

    if hasattr(torchmetrics, metric_name):
        function = getattr(torchmetrics, metric_name)
        return function(**params)
    else:
        raise ValueError(f'Wrong metric name: {metric_name}')
