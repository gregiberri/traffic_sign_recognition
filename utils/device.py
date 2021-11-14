import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def put_minibatch_to_device(minibatch):
    if DEVICE == torch.device('cuda'):
        return {key: value.to(device='cuda') if isinstance(value, torch.Tensor) else value
                for key, value in minibatch.items()}
    else:
        return minibatch
