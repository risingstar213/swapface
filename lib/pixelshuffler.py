import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import set_image_data_format, image_data_format
import tensorflow.keras.backend as K  
class PixelShuffler(Layer):
    def __init__(self, size = (2, 2), data_format = None, **kwargs):
        super().__init__(kwargs)
        if data_format == None:
            self.data_format = image_data_format()
        else:
            set_image_data_format(data_format)
            self.data_format = data_format
        self.size = tuple(size)
    def call(self, inputs):
        input_shape = inputs.shape.as_list()
        if len(input_shape) != 4:
            raise ValueError('Inputa should have 4 demensions')
        
        if self.data_format == 'channels_first':
            batch_size, c, w, h = input_shape
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
            out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
            out = K.reshape(out, (batch_size, oc, oh, ow))
        elif self.data_format == 'channels_last':
            batch_size, w, h, c= input_shape
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, oh, ow, oc))
        
        return out

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have 4 demensions')
        
        if self.data_format == 'channels_first':
            batch_size, c, w, h = input_shape
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)
            if oc * rh * rw != c:
                raise ValueError('Cannot resize the image to correspoding size.')
            return (batch_size, oc, oh, ow)
        elif self.data_format == 'channels_last':
            batch_size, w, h, c= input_shape
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)
            if oc * rh * rw != c:
                raise ValueError('Cannot resize the image to correspoding size.')
            return (batch_size, oh, ow, oc)
        
        raise ValueError('Unknown error in PixelShuffler.')
    
    def get_config(self):
        config = {'size': self.size,
                    'data_format':self.data_format}
        base_config = super().get_config()

        return dict(base_config, **config)
if __name__ == '__main__':
    layer = PixelShuffler()