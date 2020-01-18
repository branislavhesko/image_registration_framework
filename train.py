import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Configuration, Mode
from dataloaders.data_loader import get_data_loaders
from loss import HardestContrastiveLoss
from utils.timer import Timer
from utils.visualization import visualize_features, visualize_closest_points


class GeneralTrainer:

    PIXEL_LIMIT = 10
    NUM_NEG_PAIRS = 128
    NUM_NEG_INDICES_FOR_LOSS = 10

    def __init__(self, cfg: Configuration):
        self._cfg = cfg
        self._writer = SummaryWriter()
        self._model = self._cfg.MODEL(in_channels=self._cfg.IN_CHANNELS, out_channels=self._cfg.OUT_FEATURES,
                                      normalize_feature=self._cfg.NORMALIZE_FEATURES)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), momentum=self._cfg.MOMENTUM, lr=self._cfg.INITIAL_LR,
            weight_decay=self._cfg.WEIGHT_DECAY)
        self._lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self._optimizer, gamma=self._cfg.WEIGHT_DECAY)
        self._timer = Timer()
        self._data_loaders = get_data_loaders(self._cfg.DATASET, cfg, modes=self._cfg.MODES)
        self._loss = HardestContrastiveLoss(cfg)

    def train(self):
        self._model = self._model.cuda() if self._cfg.USE_CUDA else self._model
        for epoch in tqdm(range(self._cfg.EPOCHS)):
            self._timer.tic()
            self.train_single_epoch(epoch)
            self._timer.toc()

    def train_single_epoch(self, epoch):
        self._model.train()
        assert Mode.TRAIN in self._data_loaders.keys()
        for idx, inputs in tqdm(enumerate(self._data_loaders[Mode.TRAIN])):
            idx_total = idx + len(self._data_loaders[Mode.TRAIN]) * epoch
            inputs = [inp.cuda() if self._cfg.USE_CUDA else inp for inp in inputs]
            self._optimizer.zero_grad()
            output_features_1 = self._model(inputs[0])
            output_features_2 = self._model(inputs[1])
            loss = self._loss((output_features_1, output_features_2, inputs[2]))

            # self._writer.add_scalar("Loss/Positive", positive_loss.item(), idx_total)
            # self._writer.add_scalar("Loss/Negative", negative_loss.item(), idx_total)
            self._writer.add_scalar("Mean/feats1", output_features_1.detach().cpu().mean(), idx_total)
            self._writer.add_scalar("Mean/feats2", output_features_2.detach().cpu().mean(), idx_total)

            self._writer.add_scalar("Loss/Total", loss.item(), idx_total)
            loss.backward()
            self._optimizer.step()
            if not (idx % self._cfg.TRAIN_VISUALIZATION_FREQUENCY):
                feats1_reduced, feats2_reduced = visualize_features(
                    output_features_1.squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                    output_features_2.squeeze().permute([1, 2, 0]).detach().cpu().numpy())
                self._writer.add_image("Features1", feats1_reduced, idx)
                self._writer.add_image("Features2", feats2_reduced, idx)
                self._writer.add_image("Image1", inputs[0].squeeze(), idx)
                self._writer.add_image("Image2", inputs[1].squeeze(), idx)
                self._writer.add_figure("Matches", visualize_closest_points(
                    image1=inputs[0].squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                    image2=inputs[1].squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                    features1=output_features_1.squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                    features2=output_features_2.squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                    number_of_correspondences=10), idx)

    def validate(self):
        pass

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
