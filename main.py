from sys import int_info
import numpy as np
import cv2
from model.encoder import ConvertModel
from model.trainer import Trainer

def test():
    img_A = cv2.imread('img\\extract\\000076_0.jpg')
    img_B = cv2.imread('img\\extract\\000076_1.jpg')
    img_A = cv2.resize(img_A[:, :, :3], (64, 64))
    img_A = np.reshape(img_A, [-1, 64, 64, 3]).astype(np.float64) / 255
    img_B = cv2.resize(img_B[:, :, :3], (64, 64))
    img_B = np.reshape(img_B, [-1, 64, 64, 3]).astype(np.float64) / 255
    model = ConvertModel('test')
    trainer = Trainer(model, img_A, img_B, 1)
    for i in range(50):
        trainer.train_one_step()
    B = model.converter(True)(img_B)
    B = (np.reshape(B, (64, 64, 3))*255).astype(int)
    print(B.shape)
    cv2.imwrite('img\\extract\\test.jpg', B)

zoom = 1
coverage = 100
scale = 5
if __name__ == '__main__':
    img = cv2.imread('img\\extract\\000076_0.jpg')

    range_ = np.linspace(128 - coverage//2, 128 + coverage//2, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T
    print(mapx)
    mapx = mapx + np.random.normal(size=(5,5), scale=scale)
    mapy = mapy + np.random.normal(size=(5,5), scale=scale)
    print(mapx)
    interp_mapx = cv2.resize(mapx, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')
    interp_mapy = cv2.resize(mapy, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')
    print(interp_mapx)