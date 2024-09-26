from msml.core import *

ctx = Context()

A = Tensor.random(ctx, [4, 4])
B = Tensor.random(ctx, [4, 4])

R = A + B
print(R.get_op())

R = A - B
print(R.get_op())

R = A * B
print(R.get_op())

R = A / B
print(R.get_op())

R = A @ B
print(R.get_op())
