from msml.core import *

ctx = Context()

A = Tensor.full(ctx, [4, 4, 2], fill_value=2.0)
B = Tensor.full(ctx, [4, 4, 2], fill_value=2.0)
C = Tensor.full(ctx, A.shape, fill_value=2.0)

R = (A + B) * C
R.eval()
'(A + B) * C = '
R.print(True)
