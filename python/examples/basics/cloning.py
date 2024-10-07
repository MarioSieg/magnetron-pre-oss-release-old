from msml.core import *

ctx = Context()

original = Tensor.full(ctx, [4, 4], fill_value=2.0, name='original')
original.print(True)

# Create tensor with same shape and strides but don't copy any data, so data is uninitialized.
# Useful if you want to fill the tensor with data later.
# Good performance, as it doesn't allocate or copy any data.
isomorphic = original.isomorphic()
isomorphic.print(True)

# Create tensor with same shape and strides and copy data.
# Useful if you want to copy the data from another tensor.
# Worst performance, as it allocates and copies the whole data.
cloned = original.clone()
cloned.print(True)

# Create tensor with same shape and strides and reference data of original tensor.
# Useful if you want to access the data of the original tensor through different dimensions.
# Good performance, as it doesn't allocate or copy any data.
view = original.view()
view.print(True)
