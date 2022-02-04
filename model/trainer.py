import time
import numpy as np

from lib.training import TrainingDataGenerator
class Trainer():
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4
    }

    def __init__(self, model, fn_A, fn_B, batch_size, *args):
        self.batch_size = batch_size
        self.model = model
        self.generator = TrainingDataGenerator(self.random_transform_args, 160)
        self.fn_A = fn_A
        self.fn_B = fn_B

    def train_one_step(self):
        warped_A, target_A = self.generator.generate_face(self.fn_A, self.batch_size)
        warped_B, target_B = self.generator.generate_face(self.fn_B, self.batch_size)
        loss_A = self.model.autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = self.model.autoencoder_B.train_on_batch(warped_B, target_B)
        print("[# loss_A: {0:.5f}, loss_B: {1:.5f}".format(loss_A, loss_B))