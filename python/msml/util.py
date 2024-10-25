# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements utility functions for MSML. Requires matplotlib and numpy.

from msml.core import Tensor
from matplotlib import pyplot as plt
import numpy as np


def to_numpy(tensor: Tensor) -> np.array:
    buffer: list[float] = tensor.data_as_f32()
    return np.array(buffer, dtype=np.float32).reshape(tensor.shape)


def plot_tensor_as_image(tensor: Tensor, title: str | None = None) -> None:
    if title is not None:
        plt.title(title)
    interleaved = tensor.permute([2, 1, 0, 3])  # Convert from planar to interleaved shape
    src_shape = interleaved.shape
    shape = [
        src_shape[0],
        src_shape[1],
        src_shape[2]
    ]
    final = to_numpy(interleaved).reshape(shape)
    plt.imshow(final)
    plt.show()
