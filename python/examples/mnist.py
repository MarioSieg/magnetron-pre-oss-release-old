# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import os.path
import glob

from msml.core import *

IMAGES: int = 60000
IMAGE_DIM: int = 28


inputs = Tensor.empty([IMAGES, IMAGE_DIM, IMAGE_DIM])
inputs.print(True, False)
