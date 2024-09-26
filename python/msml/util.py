# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# MSML - Single header STB-style machine learning library in C99.
# MIT licensed.
# Implements utility functions for MSML. Requires matplotlib and numpy.

from msml.core import Tensor
from matplotlib import pyplot as plt
import numpy as np


def to_numpy(tensor: Tensor) -> np.array:
    return np.array(tensor.fetch_buf_data_f32(), dtype=np.float32).reshape(tensor.shape)


def plot_tensor(tensor: Tensor, title: str | None = None) -> None:
    if title is not None:
        plt.title(title)
    plt.imshow(to_numpy(tensor))
    plt.show()
