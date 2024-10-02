from msml.core import *
from msml.util import plot_tensor

ctx = Context()

# Generate a random tensor as image and plot it (width x height x channels)
R = Tensor.random(ctx, [256, 256, 3], interval=(0, 1))
plot_tensor(R, f'Noise Tensor {R.shape} - Generator: {ctx.prng_algorithm}')
