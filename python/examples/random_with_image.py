from msml import *
from matplotlib import pyplot as plt
import numpy as np

ctx = Context()

# Generate a random tensor as image and plot it (width x height x channels)
random3 = Tensor.random(ctx,[256, 256, 3], interval=(0, 1))

# Fetch the data from the device, convert to numpy array and reshape it to the original dimensions
image_data = np.array(random3.fetch_buf_data_f32(), dtype=np.float32).reshape(random3.dims)

# Plot the image to show the noise
plt.title(f'Random Tensor {random3.dims} - Generator: {ctx.prng_algorithm}')
plt.imshow(image_data)
plt.show()
