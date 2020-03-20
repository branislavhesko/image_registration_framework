from typing import List

from torch.utils.data import DataLoader

from config import Configuration, Mode
from dataloaders.dataset import ArteryVeinDataset, RetinaDatasetPop2, VesselDataset

available_datasets = {
    "ARTERY_VEIN": ArteryVeinDataset,
    "RETINA": RetinaDatasetPop2,
    "VESSEL": VesselDataset
}


def get_dataset(dataset_name):
    assert dataset_name in available_datasets.keys(), "Unknown dataset!"
    return available_datasets[dataset_name]


def get_data_loader(dataset_name, cfg: Configuration, mode: Mode):
    dataset = get_dataset(dataset_name)(cfg)
    data_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE[mode],
                             shuffle=cfg.SHUFFLE[mode], num_workers=cfg.NUM_WORKERS)
    return data_loader


def get_data_loaders(dataset_name, cfg: Configuration, modes: List):
    return {mode: get_data_loader(dataset_name, cfg, mode) for mode in modes}
