import torch


class HardestContrastiveLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config

    def forward(self, inputs):
        feats1 = inputs[0]
        feats2 = inputs[1]
        pos_pairs = inputs[2]


class PositiveHardestContrastiveLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config

    def forward(self, feats1, feats2, positive_pairs):
        feats1_selected = feats1[0, :, 0]
        feats2_selected = feats2[0, :, 1]
        return torch.mean(torch.sqrt(torch.sum(torch.pow(feats1_selected - feats2_selected, 2), dim=0)))