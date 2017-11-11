import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
from PIL import Image
import os


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    print(len(image.shape))
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def load_training_set(data_dir="training_set", output_dir="augmented_set"):
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            label = filename[0]  # first characters indicates label
            file_path = data_dir + "\\" + filename
            image = np.array(Image.open(file_path).getdata())
            image = elastic_transform(image, 34, 4)
            output_filename = label + "_0_0.png"
            try:
                scipy.misc.imsave(output_dir + "\\" + output_filename, image)
            except:
                print(filename)

if __name__ == '__main__':
    load_training_set()
