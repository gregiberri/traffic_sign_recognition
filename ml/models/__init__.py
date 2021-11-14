from torch import nn
import torchvision.models as torch_models


def get_model(model_config):
    """
    Select the model according to the model config name and its parameters

    :param model_config: model_config namespace, containing the name and the params

    :return: model
    """

    if hasattr(torch_models, model_config.name):
        function = getattr(torch_models, model_config.name)
        model = function(bool(model_config.params.pretrained))

        # change the head
        if hasattr(model, 'fc'):
            head = nn.Linear(model.fc.in_features, model_config.params.num_classes, bias=True)
            model.fc = head
        elif hasattr(model, 'classifier'):
            head = nn.Linear(model.classifier[-1].in_features, model_config.params.num_classes, bias=True)
            model.classifier[-1] = head
        else:
            raise ValueError('Can not find model last layer.')
        return model
    else:
        raise ValueError(f'Wrong model name in model configs: {model_config.name}')
