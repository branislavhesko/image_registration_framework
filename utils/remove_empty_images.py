import glob
import os

import cv2
import numpy as np

from config import VesselConfiguration

images = sorted(glob.glob(os.path.join(VesselConfiguration.PATH_TO_IMAGES_FIRST,
                                       "*." + VesselConfiguration.EXTENSION_FIRST)))
masks = sorted(glob.glob(os.path.join(VesselConfiguration.PATH_TO_MASKS,
                                      "*." + VesselConfiguration.EXTENSION_MASK)))

for image, mask in zip(images, masks):
    m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    m = m > 0

    if np.sum(m) < 0.05 * m.shape[0] * m.shape[1]:
        print("REMOVED: {}".format(image))
        os.remove(image)
        os.remove(mask)
