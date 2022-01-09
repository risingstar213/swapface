from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from lib.pixelshuffler import PixelShuffler

IMAGE_SHAPE = (64, 64, 3)
ENCOSER_DIM = 512

encoderH5 = 'encoder.h5'
decoder_AH5 = 'decoder_A.h5'
decoder_BH5 = 'decoder_B.h5'

class ConvertModel:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self):
        (face_A, face_B) = (decoder_AH5, decoder_BH5)
        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            self.decoder_A.load_weights(str(self.model_dir / face_A))
            self.decoder_B.load_weights(str(self.model_dir / face_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False
    def save_weights(self):
        self.encoder.save_weights(str(self.model_dir / encoderH5))
        self.decoder_A.save_weights(str(self.model_dir / decoder_AH5))
        self.decoder_B.save_weights(str(self.model_dir / decoder_BH5))
        print('saved model weights')
    def initModel(self):
        optimizer = Adam(lr = 5e-5, beta_1 = 0.5, beta_2 = 0.99)
        x = Input(shape =   IMAGE_SHAPE)

        output_A = self.decoder_A(self.encoder(x))
        self.autoencoder_A = Model(inputs = [x], outputs =[output_A])

        output_B = self.decoder_B(self.encoder(x))
        self.autoencoder_B = Model(inputs = [x], outputs =[output_B])

        self.autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')
    # 训练时swap = 0, 预测时swap = 1
    def converter(self, swap):
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A 
        return lambda img: autoencoder.predict(img)
    # 卷积模块,闭包便于函数式编程
    def conv(self, filters):
        def conv_block(x):
            x = Conv2D(filters, kernel_size = 5, strides = 2, padding = 'same')(x)
            x = LeakyReLU(0.1)(x) 
            return x
        return conv_block
    # 上采样模块,闭包便于函数式编程
    def upscale(self, filters):
        def upscale_block(x):
            x = Conv2D(filters * 4, knernel_size = 3, padding = 'same')
            x = LeakyReLU(0.1)(x) 
            x = PixelShuffler()(x)
            return x
        return upscale_block
    # 三层卷积 + 全连接
    # 修改为FCN？
    def Encoder(self):
        input = Input(shape = IMAGE_SHAPE)
        x = input
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)

        x = Dense(ENCOSER_DIM)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((-1, 4, 4, 1024))(x)
        x = self.upscale(512)(x)

        return Model(inputs = [input], outputs = [x])
    
    def Decoder(self):
        input = Input(shape = (8, 8, 512))
        x = input
        x = self.upscale(256)(x)
        x = self.upscale(128)(x)
        x = self.upscale(64)(x)
        x = Conv2D(3, kernel_size = 5, padding='same', activation = 'sigmoid')(x)
        return Model(inputs = [input], outputs = [x])