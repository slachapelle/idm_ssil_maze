import numpy as np

from skimage.draw import random_shapes


def random_shape_maze(width, height, max_shapes, max_size, allow_overlap, shape=None, random_state=None):
    x, _ = random_shapes([height, width], max_shapes, max_size=max_size, channel_axis=None, shape=shape,
                         allow_overlap=allow_overlap, random_seed=random_state)
    
    x[x == 255] = 0
    x[np.nonzero(x)] = 1
    
    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1
    
    return x
