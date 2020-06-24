import torch


class SegmentationBlock(torch.nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self._block = self._get_block(in_channels=in_channels, out_channels=(in_channels - num_classes) // 2)
        self._final_block = self._get_final_block((in_channels - num_classes + 1) // 2, num_classes)

    def _get_block(self, in_channels, out_channels):
        block = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        ])
        return block

    def _get_final_block(self, in_channels, num_classes):
        final = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1)
        ])
        return final

    def forward(self, input_):
        out = self._block(input_)
        out = self._final_block(out)
        return out
