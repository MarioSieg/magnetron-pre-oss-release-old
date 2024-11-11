# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements utility functions for WAVELET. Requires matplotlib and numpy.

from wavelet.core import Tensor
from matplotlib import pyplot as plt
import numpy as np


def to_numpy(tensor: Tensor, deinterleave: bool = False) -> np.array:
    if deinterleave:
        w: int = tensor.width
        h: int = tensor.height
        c: int = tensor.channels
        data = np.array(tensor.data_as_f32()).reshape(c, h, w)
        data = np.transpose(data, (1, 2, 0))
        return (data * 255.0).astype(np.uint8)
    else:
        return np.array(tensor.data_as_f32(), dtype=np.float32).reshape(tensor.shape)


def plot_tensor_image(tensor: Tensor, title: str | None = None) -> None:
    if title is not None:
        plt.title(title)
    plt.imshow(to_numpy(tensor, deinterleave=True))
    plt.show()


def plot_tensor_scatter_3d(tensor: Tensor, title: str | None = None) -> None:
    data = to_numpy(tensor)
    x, y, z = np.where(data > 0)
    fig = plt.figure()
    if title is not None:
        plt.title(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='o')
    plt.show()
