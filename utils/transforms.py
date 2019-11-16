import torch


def flatten(pair_indices, num_cols):
    """
    Pair indices are in form [N x 2], with [row, column form]
    :param pair_indices: -
    :param num_cols: number of columns in matrix
    :return:
    """
    return pair_indices[:, 0] * num_cols + pair_indices[:, 1]


def unflatten(pair_indices_flat, num_cols):
    return torch.cat([pair_indices_flat // num_cols, pair_indices_flat % num_cols], dim=1)