from msml.core import *

ctx = Context()

original = Tensor.full(ctx, [4, 1], fill_value=2.0, name='original')
print(original.is_transposed)

tranposed = original.transpose()()
print(original.is_transposed)

original.print(True)
tranposed.print(True)
