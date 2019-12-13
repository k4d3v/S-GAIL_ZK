import numpy as np
import torch
import math


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def min_max(x, axis=None):
    """
    Normalization
    @param x:
    @param axis:
    @return:
    """
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result


def delete(x, s_min, s_max):
    """
    Reduce and clip x
    @param x:
    @param s_min:
    @param s_max:
    @return:
    """
    y = np.delete(x, [4, 5, 8, 9, 10])
    result = np.clip((y - s_min) / (s_max - s_min), 0.0, 1.0)
    return result


def norm_act(x, a_min, a_max):
    """
    Normalize action
    @param x:
    @param a_min:
    @param a_max:
    @return:
    """
    result = np.clip((x - a_min) / (a_max - a_min), 0.0, 1.0)
    return result
