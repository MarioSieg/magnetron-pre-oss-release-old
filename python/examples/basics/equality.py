from msml.core import *

ctx = Context()

# A and B are equal
# A and C are not equal
A = Tensor.random(ctx, [4, 4, 2])
B = A.clone()()
C = Tensor.random(ctx, [4, 4, 2])

# Check for strict equality (floating point errors might occur)
print(A == B)  # Compare A and B (shape, data, etc.)
print(A == C)  # Compare A and C (shape, data, etc.)
assert A == B
assert A != C

# Check for equality within a certain tolerance (epsilon)
# is_close() returns a tuple of (bool, float) where the float is the percentage of equal elements
print(A.is_close(B))  # Compare A and B within epsilon (shape, data, etc.)
print(A.is_close(C))  # Compare A and C within epsilon (shape, data, etc.)

# Same but print differences
A.is_close(B, print_eq_percent=True)
A.is_close(C, print_eq_percent=True)

