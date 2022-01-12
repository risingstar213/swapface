import time
import numpy as np


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
        self.target_A = fn_A
        self.target_B = fn_B

    def train_one_step(self):
        loss_A = self.model.autoencoder_A.train_on_batch(self.target_A, self.target_A)
        loss_B = self.model.autoencoder_B.train_on_batch(self.target_B, self.target_B)
        print("[# loss_A: {0:.5f}, loss_B: {1:.5f}".format(loss_A, loss_B))