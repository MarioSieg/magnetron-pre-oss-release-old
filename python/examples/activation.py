from msml.core import *

ctx = Context()

A = Tensor.full(ctx, [4, 4], fill_value=2.0)
A.print(True)
R = A.softmax()
R().print(True)
