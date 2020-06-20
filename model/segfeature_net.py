import torch

from config import Configuration
from model.resunet import ResUNetBN2C
from model.segmentation_block import SegmentationBlock


class SegFeatureNet(torch.nn.Module):

    def __init__(self, config: Configuration):
        super().__init__()
        self._feature_net = ResUNetBN2C(
            in_channels=config.IN_CHANNELS, normalize_feature=config.NORMALIZE_FEATURES,
            out_channels=config.OUT_FEATURES)
        self._seg_net = SegmentationBlock(in_channels=config.OUT_FEATURES, num_classes=config.NUM_CLASSES)

    def forward(self, input_):
        feats = self._feature_net(input_)
        return feats, self._seg_net(feats)
