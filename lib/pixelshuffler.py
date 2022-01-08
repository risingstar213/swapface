from tensorflow.keras.layers import Layer


class PixelShuffler(Layer):
    def __init__(self, size = (2, 2), data_format = None, **kwargs):
        super().__init__(kwargs)