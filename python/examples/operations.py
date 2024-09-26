from msml.core import *

ctx = Context()

A = Tensor.random(ctx,[4, 4])
B = Tensor.random(ctx,[4, 4])
R = Tensor.isomorphic_clone(A)
R.set_op_with_args(Operation.ADD, A, B)
print(R.get_op())
print(R.get_arg(0))
print(R.get_arg(1))
