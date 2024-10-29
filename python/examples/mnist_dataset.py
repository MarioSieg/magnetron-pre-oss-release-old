# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>


"""
Clone MNIST jpeg data:
https://www.kaggle.com/datasets/scolianni/mnistasjpg?resource=download
into current directory and unpack it.
File
"""

from os import path

from msml.core import *
from glob import glob
from cv2 import imread

IMAGES: int = 42000  # should be 60k fuckers
IMAGE_DIM: int = 28
TRAINING_SET_DIR: str = '../datasets/MNIST/training_set'

data: list[float] = []
num_images = 0
for i in range(0, 10):
    source_dir: str = path.join(TRAINING_SET_DIR, str(i))
    assert path.exists(source_dir)
    for file in glob(source_dir + '/**.jpg'):
        print(f'Reading file {file}')
        img: list[float] = (imread(file).astype(float) / 255.0).flatten().tolist()
        assert len(img) == 28 * 28 * 3
        data.extend(img)
        num_images += 1
total: int = num_images * IMAGE_DIM * IMAGE_DIM * 3
assert len(data) == total, f'Expected {total} elements, got {len(data)}'
print(f'Processed {num_images} images, total elems {len(data)}, storing to MSML storage database')

Tensor.const(data=data, shape=[num_images, IMAGE_DIM, IMAGE_DIM, 3]).save('mnist.msml')
