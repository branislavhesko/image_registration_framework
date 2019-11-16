import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Configuration, Mode
from dataloaders.data_loader import get_data_loaders
from utils.average_meter import AverageMeter
from utils.timer import Timer


class GeneralTrainer:

    PIXEL_LIMIT = 10
    NUM_NEG_PAIRS = 4096

    def __init__(self, cfg: Configuration):
        self._cfg = cfg
        self._writer = SummaryWriter(log_dir="runs/retina")
        self._model = self._cfg.MODEL(in_channels=self._cfg.IN_CHANNELS, out_channels=self._cfg.OUT_FEATURES,
                                      normalize_feature=self._cfg.NORMALIZE_FEATURES)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), momentum=self._cfg.MOMENTUM, lr=self._cfg.INITIAL_LR,
            weight_decay=self._cfg.WEIGHT_DECAY)
        self._lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self._optimizer, gamma=self._cfg.WEIGHT_DECAY)
        self._timer = Timer()
        self._data_loaders = get_data_loaders(self._cfg.DATASET, cfg, modes=self._cfg.MODES)

    def train(self):
        for epoch in tqdm(range(self._cfg.EPOCHS)):
            self._timer.tic()
            self.train_single_epoch(epoch)
            self._timer.toc()

    def train_single_epoch(self, epoch):
        assert Mode.TRAIN in self._data_loaders.keys()
        average_meter = AverageMeter()

        for idx, inputs in enumerate(self._data_loaders[Mode.TRAIN]):
            inputs = [inp.cuda() if self._cfg.USE_CUDA else inp for inp in inputs]
            self._optimizer.zero_grad()
            output_features_1 = self._model(inputs[0])
            output_features_2 = self._model(inputs[1])

            loss = self.loss_function(output_features_1, output_features_2, inputs[2])
            loss.backward()
            self._optimizer.step()

    def validate(self):
        pass

    def loss_function(self, features1, features2, pos_indices):
        negative_loss = self._negative_loss(features1, features2, pos_indices)
        positive_loss = self._positive_loss(features1, features2, pos_indices)
        return positive_loss - self._cfg.NEGATIVE_LOSS_COEF * negative_loss

    def _positive_loss(self, features1, features2, indices):
        features1_pos = torch.index_select(features1.view(*features1.shape[:2], -1), 2, indices[:, 0])
        features2_pos = torch.index_select(features2.view(*features2.shape[:2], -1), 2, indices[:, 1])
        return torch.sqrt(torch.sum(torch.pow(features1_pos - features2_pos, 2), dim=1))

    def _negative_loss(self, feats1, feats2, pos_pairs):
        B, C, H, W = feats1.shape
        feats1_flat = feats1.view(C, -1)
        negative_loss = 0
        negative_indices = np.random.choice(feats1_flat.shape[-1], self.NUM_NEG_PAIRS)
        for negative_idx in negative_indices:
            correspondence = pos_pairs[torch.abs(pos_pairs[:, 0] - negative_idx).argmin(), 1]
            row = correspondence // W
            col = correspondence % W
            mask = torch.ones_like(feats2)
            mask[:, :, row - self.PIXEL_LIMIT: row + self.PIXEL_LIMIT,
                 col - self.PIXEL_LIMIT, col + self.PIXEL_LIMIT] = 0
            feats2_masked = feats2[mask > 0].view(C, -1)
            dist = torch.sqrt(torch.sum(torch.pow(feats1[:, negative_idx] - feats2_masked, 2), dim=1))
            negative_loss += dist.min().item()

        return negative_loss

    def load_checkpoint(self):
        pass

    def save_checkpoint(self, file_name):
        state_dict = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "learning_rate": self._lr_scheduler.state_dict()
        }
        if not os.path.exists(self._cfg.CHECKPOINT_PATH):
            os.makedirs(self._cfg.CHECKPOINT_PATH)
        torch.save(state_dict, os.path.join(self._cfg.CHECKPOINT_PATH, file_name + ".pth"))
