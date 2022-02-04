from asyncio import wrap_future
from sys import int_info
import numpy as np
import cv2
from model.encoder import ConvertModel
from model.trainer import Trainer
from lib.training import TrainingDataGenerator

def test():
    '''
    img_A = cv2.imread('img\\extract\\000076_0.jpg')
    img_B = cv2.imread('img\\extract\\000076_1.jpg')
    img_A = cv2.resize(img_A[:, :, :3], (64, 64))
    img_A = np.reshape(img_A, [-1, 64, 64, 3]).astype(np.float64) / 255
    img_B = cv2.resize(img_B[:, :, :3], (64, 64))
    img_B = np.reshape(img_B, [-1, 64, 64, 3]).astype(np.float64) / 255
    '''
    fn_A = 'img\\extract\\000076_0.jpg'
    fn_B = 'img\\extract\\000076_1.jpg'
    model = ConvertModel('test')
    trainer = Trainer(model, fn_A, fn_B, 5)
    for i in range(1000):
        trainer.train_one_step()
    
    img_B = cv2.imread(fn_B)
    img_B = cv2.resize(img_B[:, :, :3], (64, 64))
    img_B = np.reshape(img_B, [-1, 64, 64, 3]).astype(np.float64) / 255

    B = model.converter(True)(img_B)
    B = (np.reshape(B, (64, 64, 3))*255).astype(int)

    cv2.imwrite('img\\extract\\test.jpg', B)

zoom = 1
coverage = 100
scale = 5
if __name__ == '__main__':
    test()
    '''
    fn_A = 'img\\extract\\000076_0.jpg'
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4
    }
    generator = TrainingDataGenerator(random_transform_args, 160)
    image = generator.color_adjust(cv2.imread(fn_A))
    
    wrap_A, target_A = generator.transform_image(image)
    wrap_A = (np.reshape(wrap_A, (64, 64, 3))*255).astype(int)
    target_A = (np.reshape(target_A, (64, 64, 3))*255).astype(int)
    
    cv2.imwrite('test1.jpg', wrap_A)
    cv2.imwrite('test2.jpg', target_A)
    '''