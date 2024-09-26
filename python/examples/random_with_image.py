from msml.core import *
from msml.util import plot_tensor

ctx = Context()

# Generate a random tensor as image and plot it (width x height x channels)
R = Tensor.random(ctx, [256*8, 256*8, 3], interval=(0, 1))
X = Tensor.isomorphic_clone(R)
X.fill(1.0)
img = R * X
plot_tensor(img, f'Noise Tensor {img.shape} - Generator: {ctx.prng_algorithm}')
