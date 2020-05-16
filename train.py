import os
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Configuration, Mode
from dataloaders.data_loader import get_data_loaders
from loss import ArteryVeinLoss, HardestContrastiveLoss
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
            weight_decay=self._cfg.WEIGHT_DECAY, nesterov=True)
        self._lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self._optimizer, gamma=self._cfg.LR_DECAY)
        self._timer = Timer()
        self._data_loaders = get_data_loaders(self._cfg.DATASET, cfg, modes=self._cfg.MODES)
        self._loss = HardestContrastiveLoss(cfg)

    def train(self):
        self._model = self._model.cuda() if self._cfg.USE_CUDA else self._model
        print(self._model)
        for epoch in tqdm(range(self._cfg.EPOCHS)):
            self._timer.tic()
            self.train_single_epoch(epoch)
            self._lr_scheduler.step(epoch=epoch)
            self._timer.toc()

    def train_single_epoch(self, epoch):
        self._model.train()
        assert Mode.TRAIN in self._data_loaders.keys()
        for idx, inputs in tqdm(enumerate(self._data_loaders[Mode.TRAIN])):
            idx_total = idx + len(self._data_loaders[Mode.TRAIN]) * epoch
            inputs = [inp.cuda() if self._cfg.USE_CUDA else inp for inp in inputs]
            self._optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.calculate_loss(outputs, inputs)
            # self._writer.add_scalar("Loss/Positive", positive_loss.item(), idx_total)
            # self._writer.add_scalar("Loss/Negative", negative_loss.item(), idx_total)
            loss.backward()
            print("Loss: {}".format(loss.item()))

            self._optimizer.step()
            self.visualize(inputs, outputs, loss, idx, idx_total, epoch)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def visualize(self, *args, **kwargs):
        pass

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


class RegistrationTrainer(GeneralTrainer):

    def forward(self, inputs):
        output_features_1 = self._model(inputs[0])
        output_features_2 = self._model(inputs[1])
        return [output_features_1, output_features_2]

    def calculate_loss(self, outputs, inputs):
        return self._loss((outputs[0], outputs[1], inputs[2]))

    def visualize(self, inputs, outputs, loss, idx, idx_total, epoch=None):
        self._writer.add_scalar("Mean/feats1", outputs[0].detach().cpu().mean(), idx_total)
        self._writer.add_scalar("Mean/feats2", outputs[1].detach().cpu().mean(), idx_total)

        self._writer.add_scalar("Loss/Total", loss.item(), idx_total)
        if not (idx % self._cfg.TRAIN_VISUALIZATION_FREQUENCY):
            feats1_reduced, feats2_reduced = visualize_features([
                outputs[0].squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                outputs[1].squeeze().permute([1, 2, 0]).detach().cpu().numpy()])
            self._writer.add_image("Features1", torch.from_numpy(feats1_reduced).permute([2, 0, 1]), idx)
            self._writer.add_image("Features2", torch.from_numpy(feats2_reduced).permute([2, 0, 1]), idx)
            self._writer.add_image("Image1", inputs[0].squeeze(), idx)
            self._writer.add_image("Image2", inputs[1].squeeze(), idx)
            self._writer.add_figure("Matches", visualize_closest_points(
                image1=inputs[0].squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                image2=inputs[1].squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                features1=outputs[0].squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                features2=outputs[1].squeeze().permute([1, 2, 0]).detach().cpu().numpy(),
                number_of_correspondences=10), idx)


class VesselTrainer(GeneralTrainer):

    def forward(self, inputs):
        image, mask = inputs
        return self._model(image)

    def calculate_loss(self, outputs, inputs):
        return self._loss(outputs, inputs[1])

    def visualize(self, inputs, outputs, loss, idx, idx_total, *args):
        self._writer.add_scalar("Mean/feats1", outputs[0].detach().cpu().mean(), idx_total)

        self._writer.add_scalar("Loss/Total", loss.item(), idx_total)
        if not (idx % self._cfg.TRAIN_VISUALIZATION_FREQUENCY):
            feats = outputs[0].squeeze().permute([1, 2, 0]).detach().cpu().numpy()
            # TODO: refactor to a method taking an iterable!
            feats1_reduced = visualize_features([feats])
            self._writer.add_image("Features", torch.from_numpy(feats1_reduced[0]).permute([2, 0, 1]), idx)
            self._writer.add_image("Image1", inputs[0].squeeze(), idx)


class VeinArteryTrainer(GeneralTrainer):

    def __init__(self, config):
        super().__init__(config)
        self._loss = ArteryVeinLoss(config)

    def forward(self, inputs):
        image, mask = inputs
        return self._model(image)

    def calculate_loss(self, outputs, inputs):
        image, mask = inputs
        return self._loss(outputs, mask)

    def visualize(self, inputs, outputs, loss, idx, idx_total, epoch=0):
        self._writer.add_scalar("Mean/feats1", outputs[0].detach().cpu().mean(), idx_total)
        self._writer.add_scalar("Loss/Total", loss.item(), idx_total)
        mask = inputs[1].squeeze()
        mask_colored = torch.zeros(3, *mask.shape, dtype=torch.float)
        mask_colored[0, mask == 1] = 1.
        mask_colored[1, mask == 2] = 1.
        mask_colored[2, mask == 3] = 1.
        self._writer.add_image(f"Mask/{idx}", mask_colored, epoch)
        self._writer.add_image(f"Original_image/{idx}", inputs[0].squeeze(), epoch)

        feats = outputs.squeeze().permute([1, 2, 0]).detach().cpu().numpy()
        feats_pca = visualize_features([feats])[0]
        self._writer.add_image(f"Features/{idx}", torch.from_numpy(feats_pca).permute([2, 0, 1]), epoch)
        feats_pca_mask = visualize_features([feats[mask.cpu().numpy() > 0.5, :]])[0]
        output_mask = np.zeros_like(feats_pca)
        output_mask[mask.cpu().numpy() > 0.5, :] = np.squeeze(feats_pca_mask)
        self._writer.add_image(f"Features_masked/{idx}", torch.from_numpy(output_mask).permute([2, 0, 1]), epoch)
