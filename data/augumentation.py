import numpy as np
import scipy.misc
from PIL import Image, ImageOps
import os
from imgaug import augmenters as iaa
import shutil

def copy_to_folders():
    for i in range(10):
        dir = "augmented_set\\" + str(i)
        try:
            os.mkdir(dir)
        except:
            files = os.listdir(dir)
            for file in files:
                os.remove(os.path.join(dir, file))
    for root, dirs, files in os.walk("training_set"):
        for filename in files:
            shutil.copy("training_set\\" + filename, "augmented_set\\" + filename[0] + "\\" + filename)


seq = iaa.Sequential([
    # iaa.Scale(0.8, "nearest"),
    iaa.CropAndPad(px=((0, 5), (0, 5), (0, 3), (0, 3)))
])

if __name__ == '__main__':
    # copy_to_folders()
    data_dir = "training_set"
    output_dir = "augmented_set"
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            label = filename[0]  # first characters indicates label
            file_path = data_dir + "\\" + filename
            image = Image.open(file_path).convert("L")
            image = ImageOps.invert(image)
            image = np.matrix(image)
            # print(image)
            for i in range(1):
                augmented_image = seq.augment_image(image)
                augmented_image = np.invert(augmented_image)
                output_filename = label + filename + str(i) + ".png"
                try:
                    scipy.misc.imsave(output_dir + "\\" + output_filename, augmented_image)
                except:
                    print(filename)