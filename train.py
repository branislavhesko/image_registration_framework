import os

import numpy as np
import torch
import torch.nn.functional as F
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
            print(output_features_2.abs().mean())
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

    def loss_function(self, features1, features2, pos_indices):
        positive_loss = self._positive_loss(features1, features2, pos_indices)
        negative_loss = self._negative_loss(features1, features2, pos_indices)
        return positive_loss, negative_loss

    def _positive_loss(self, features1, features2, indices):
        B, C, H, W = features1.shape
        feats1_flat = features1.view(C, -1)
        feats2_flat = features2.view(C, -1)

        features1_pos = feats1_flat[:, indices[0, :, 0].long()]
        features2_pos = feats2_flat[:, indices[0, :, 1].long()]

        return torch.mean(F.relu(torch.sum(torch.pow(features1_pos - features2_pos, 2), dim=0)))

    def _negative_loss(self, feats1, feats2, pos_pairs):
        B, C, H, W = feats1.shape
        feats1_flat = feats1.view(C, -1)
        feats2_flat = feats2.view(C, -1)

        negative_indices = np.random.choice(feats1_flat.shape[-1], self.NUM_NEG_PAIRS, replace=False)
        negative_loss = torch.empty(self.NUM_NEG_PAIRS)
        for index, negative_idx in enumerate(negative_indices):
            negative_indices_hard = torch.empty(self.NUM_NEG_INDICES_FOR_LOSS)
            mask = np.array([np.arange(negative_idx + W * (i - self.PIXEL_LIMIT) - self.PIXEL_LIMIT,
                              negative_idx + W * (i - self.PIXEL_LIMIT) + self.PIXEL_LIMIT)
                    for i in range(self.PIXEL_LIMIT * 2)]).reshape(1, -1)
            mask = mask[(mask > 0) & (mask < feats1_flat.shape[-1])]
            dist = F.relu(torch.sum(torch.pow(feats1_flat[
                                                  :, negative_idx].unsqueeze(dim=1) - feats2_flat, 2), dim=0))
            dist_sorted_indices = dist.argsort()

            idx = 0
            counter = 0
            while counter < self.NUM_NEG_INDICES_FOR_LOSS and idx < dist.shape[-1]:
                if not dist[dist_sorted_indices[idx]].item() in mask:
                    negative_indices_hard[counter] = -dist[dist_sorted_indices[idx]]
                    counter += 1
                    mask_new = np.array([np.arange(dist_sorted_indices[idx].item() + W * (i - self.PIXEL_LIMIT) - self.PIXEL_LIMIT,
                                                   dist_sorted_indices[idx].item() + W * (i - self.PIXEL_LIMIT) + self.PIXEL_LIMIT)
                                     for i in range(self.PIXEL_LIMIT * 2)]).reshape(1, -1)
                    mask_new = mask_new[(mask_new > 0) & (mask_new < feats1_flat.shape[-1])]
                    mask = np.concatenate([mask, mask_new], axis=0)
                idx += 1

            negative_loss[index] = torch.mean(negative_indices_hard)

        return torch.mean(negative_loss.cuda())

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
