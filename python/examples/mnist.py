# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import wavelet.core as wl
from wavelet.models import SequentialModel, DenseLayer
import matplotlib.pyplot as plt

from wavelet.util import plot_tensor_as_image

EPOCHS: int = 10
LEARNING_RATE: float = 1e-3

# Load MNIST dataset
images: wl.Tensor = wl.Tensor.load('mnist_images.wavelet')

network = SequentialModel([
    DenseLayer(784, 250),
    DenseLayer(250, 100),
    DenseLayer(100, 10)
])

network.train([images], [images.transpose().clone()], EPOCHS, LEARNING_RATE)

network.summary()
