"""Dataloaders for lodopab CT dataset"""
from os import path, PathLike
import typing
from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset
from dival.util.torch_utility import RandomAccessTorchDataset
from torch.utils.data import DataLoader
from musc import DATASET_DIR


def get_dataloaders_ct(batch_size: int = 1,
                       num_workers: int = 0,
                       cache_dir: typing.Union[str, PathLike] = path.join(
                           DATASET_DIR, 'cache_lodopab'),
                       include_validation: bool = True,
                       train_percent: int = 100,
                       validation_len: int = 100,
                       test_len: int = 100
                       ):
    """Construct pytorch loader for the LoDoPaB CT dataset.

    This function follows the logic here:
    https://github.com/jleuschn/dival/blob/master/dival/examples/ct_train_fbpunet.py#L25-L43

    Args:
        batch_size (int, optional): batch size for training.
            Defaults to 1.
        num_workers (int, optional): Defaults to 0.
        cache_dir (_type_, optional): Defaults to
            path.join(DATASET_DIR, 'cache_lodopab').
        include_validation (bool, optional): Defaults to True.
        train_percent (int, optional): Defaults to 100.
        validation_len (int, optional): Defaults to 100.
        test_len (int, optional): Defaults to 100.


    Returns:
        dataloaders: dataloaders for training, validation, and test.
    """
    if include_validation:
        parts = ['train', 'validation', 'test']
        batch_sizes = {'train': batch_size, 'validation': 1, 'test': 1}

    else:
        parts = ['train', 'test']
        batch_sizes = {'train': batch_size, 'test': 1}

    cache_files = {
        part: (path.join(cache_dir,
                         'cache_lodopab_' + part + '_fbp.npy'), None)
        for part in parts
    }

    standard_dataset = get_standard_dataset('lodopab', impl='astra_cuda')
    ray_trafo = standard_dataset.get_ray_trafo(impl='astra_cuda')
    dataset = get_cached_fbp_dataset(standard_dataset, ray_trafo, cache_files)

    dataset.train_len = int(dataset.train_len * train_percent / 100)

    print('train percent: ', train_percent)
    print('train dataset len: ', dataset.train_len)
    dataset.validation_len = validation_len
    dataset.test_len = test_len

    # create PyTorch datasets
    datasets = {
        x: RandomAccessTorchDataset(dataset=dataset,
                                    part=x,
                                    reshape=((1, ) + dataset.space[0].shape,
                                             (1, ) + dataset.space[1].shape))
        for x in parts
    }

    dataloaders = {
        x: DataLoader(datasets[x],
                      batch_size=batch_sizes[x],
                      pin_memory=True,
                      shuffle=(x == 'train'),
                      num_workers=num_workers)
        for x in parts
    }

    return dataloaders
