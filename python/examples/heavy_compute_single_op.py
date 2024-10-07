import time

from msml.core import *
from msml.util import plot_tensor

ctx = Context()

# Generate a random tensor as image and plot it (width x height x channels)
R = Tensor.random(ctx, [16384, 16384, 3], interval=(0, 1), name='Random Noise')

X = Tensor.isomorphic(R)
X.name = 'Coefficient'
X.fill(3)

img = R * X

now: int = time.time_ns()

img() # Evaluate the tensor

time_took: float = (time.time_ns() - now) / 1_000_000_000

print(f'Elements: {img.num_elements}')
R.print(False)
print(f'Evaluated in {time_took:.6f} S')
