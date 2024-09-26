import time

from msml.core import *
from msml.util import plot_tensor

ctx = Context()

# Generate a random tensor as image and plot it (width x height x channels)
R = Tensor.random(ctx, [4096, 4096, 3], interval=(0, 1))
X = Tensor.isomorphic_clone(R)
X.fill(1)
img = R * X
now = time.time_ns()
img.eval()
time_took = time.time_ns() - now
print(f'Evaluated in {time_took} ns')
plot_tensor(img, f'Noise Tensor {img.shape} - Generator: {ctx.prng_algorithm}')
