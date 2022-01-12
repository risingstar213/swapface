import numpy as np
import cv2
from numpy.core.defchararray import rpartition
from tensorflow.python.keras.backend import sigmoid

class TrainingDataGenerator():
    def __init__(self, random_transform_args, coverge, scale, zoom):
        self.random_transform_args = random_transform_args
        self.coverge = coverge
        self.scale = scale
        self.zoom = zoom
    
    def color_adjust(self, img):
        return img / 255.0

    def random_transform(self, image, rotation_range, 
        zoom_range, shift_range, random_flip):
        h, w = image.shape[0:2]
        rotation = np.random.uniform(-rotation_range, rotation_range)
        scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        dw = np.random.uniform(-shift_range, shift_range) * w
        dh = np.random.uniform(-shift_range, shift_range) * h

        mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
        mat[:, 2] += (dw, dh)
        result = cv2.warpAffine(
            image, (w, h), borderMode=cv2.BORDER_REPLICATE
        )
        # 左右翻转
        if np.random.random() < random_flip:
            result = result[:, ::-1]
        return result

    def radom_warp(self, image, coverge, scale = 5, zoom = 1):
        assert image.shape == (256, 256, 3)
        change_range = np.linspace(128 - coverge // 2, 128 + coverge // 2, 5)
        map_x = np.broadcast_to(change_range, (5, 5))
        map_y = np.transpose(map_x)

        map_x += np.ramdom.normal(size = (5, 5), scale = scale)
        map_y += np.ramdom.normal(size = (5, 5), scale = scale)