from tensorflow.keras.layers import Conv2D, Dense, Input, LeakyReLU

IMAGE_SHAPE = (64, 64, 3)

class ConvertModel:
    def __init__(self, model_dir):
        self.model_dir = model_dir
    


    def Encoder(self):
        x = Input()
