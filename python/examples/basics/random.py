from msml.core import *

ctx = Context()

# Generate a 4x4 tensor filled with random values between 0 and 1
random1 = Tensor.random(ctx, [4, 4], interval=(0, 1))
random1.print(True)

# Generate a 10x1 tensor filled with random values between -10 and 10
random2 = Tensor.random(ctx, [10], interval=(-10, 10))
random2.print(True)
