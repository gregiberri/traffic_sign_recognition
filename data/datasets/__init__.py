from torch.utils.data import DataLoader

from data.datasets.traffic_sign_dataset import TrafficSignDataloader


def get_dataloader(data_config, mode):
    # get the iterator object
    if data_config.name == 'traffic_signs':
        dataset = TrafficSignDataloader(data_config.params, mode)
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    # batch size 1 for validation
    batch_size = int(data_config.params.batch_size) if 'train' in mode else 1

    # make the torch dataloader object
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=data_config.params.workers,
                        drop_last=True,
                        shuffle='train' in mode,
                        pin_memory=data_config.params.load_into_memory)

    return loader
