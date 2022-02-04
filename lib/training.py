import numpy as np
import cv2
import random
from lib.alignment import umeyama

class TrainingDataGenerator():
    def __init__(self, random_transform_args, coverge, scale=5, zoom=1):
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
            image,  mat, (w, h), borderMode=cv2.BORDER_REPLICATE
        )
        # 左右翻转
        if np.random.random() < random_flip:
            result = result[:, ::-1]
        return result

    def random_warp(self, image, coverge, scale = 5, zoom = 1):
        assert image.shape == (256, 256, 3)
        change_range = np.linspace(128 - coverge // 2, 128 + coverge // 2, 5)
        map_x = np.broadcast_to(change_range, (5, 5))
        map_y = np.transpose(map_x)

        map_x = map_x + np.random.normal(size = (5, 5), scale = scale)
        map_y = map_y + np.random.normal(size = (5, 5), scale = scale)

        expand_mapx = cv2.resize(map_x, (80 * zoom, 80 * zoom))[8*zoom:72*zoom, 8*zoom:72*zoom].astype('float32')
        expand_mapy = cv2.resize(map_y, (80 * zoom, 80 * zoom))[8*zoom:72*zoom, 8*zoom:72*zoom].astype('float32')
        
        # 线性插值
        warped_image = cv2.remap(image, expand_mapx, expand_mapy, cv2.INTER_LINEAR)
    
        src_points = np.stack([map_x.ravel(), map_y.ravel()], axis = -1)
        dst_points = np.mgrid[0:65*zoom:16*zoom,0:65*zoom:16*zoom].T.reshape(-1,2)
        mat = umeyama(src_points, dst_points, True)[0:2]

        target_image = cv2.warpAffine(image, mat, (64 * zoom, 64 * zoom))

        return warped_image, target_image

    def transform_image(self, image):
        
        image = cv2.resize(image, (256,256))
        image = self.random_transform( image, **self.random_transform_args )
        warped_img, target_img = self.random_warp( image, self.coverge, self.scale, self.zoom )
        
        return warped_img, target_img

    def generate_face(self, fn, batch_size):
        try:
            image = self.color_adjust(cv2.imread(fn))
        except TypeError:
            raise Exception("Error while reading image", fn)

        rtn = np.float32([self.transform_image(image) for i in range(batch_size)])
        return (rtn[:, 0, :, :, :], rtn[:, 1, :, :, :])