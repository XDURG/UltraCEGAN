from __future__ import print_function, division
from metrics import ssim_metric, psnr_metric
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from data_loader import DataLoader
import numpy as np
from utils import ssim_loss, color_histogram_loss

class CycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.dataset_name = 'sythetic_ceus'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        self.gf = 32
        self.df = 64
        self.lambda_gan = 1.0
        self.lambda_cycle = .25
        self.lambda_id = .25
        self.lambda_ssim = .5
        self.lambda_brightness = .25
        optimizer = Adam(0.001, 0.5)
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.g_AB = self.build_generator()
        self.g_AB.load_weights('./unet_seg_72_100.hdf5')
        self.g_BA = self.build_generator()
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)
        self.d_A.trainable = False
        self.d_B.trainable = False
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id, fake_B])
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                                    'mae', 'mae',
                                    'mae', 'mae', [ssim_loss, color_histogram_loss]],
                            loss_weights=[  self.lambda_gan, self.lambda_gan,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id, [self.lambda_ssim, self.lambda_brightness]],
                            metrics=['accuracy', 'accuracy', ssim_metric, ssim_metric, ssim_metric, ssim_metric, [ssim_metric, psnr_metric]],
                            optimizer=optimizer)

    def build_generator(self):
        # U-Net Generator
        def conv2d(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*16)
        u1_0 = deconv2d(d5, d4, self.gf*8)
        u1 = deconv2d(u1_0, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)
        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        return Model(d0, output_img)

    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d
        img = Input(shape=self.img_shape)
        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch_pair(batch_size)):
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)
                self.d_A.train_on_batch(imgs_A, valid)
                self.d_A.train_on_batch(fake_A, fake)
                self.d_B.train_on_batch(imgs_B, valid)
                self.d_B.train_on_batch(fake_B, fake)
                for n_gen in range(2):
                    self.combined.train_on_batch([imgs_A, imgs_B],
                                                 [valid, valid,
                                                  imgs_A, imgs_B,
                                                  imgs_A, imgs_B, imgs_B])
                if batch_i % sample_interval == 0:
                    self.save_model(epoch, batch_i)

    def save_model(self, epoch, batch_i):
        def save(model, model_name):
            model_path = "./sythetic_ceus_%s_%s_%s.json" % (model_name, epoch, batch_i)
            weights_path = "./sythetic_ceus_%s_%s_%s_weights.hdf5" % (model_name, epoch, batch_i)
            open(model_path, 'w').write(model.to_json())
            model.save_weights(weights_path)
        save(self.g_AB, "generatorAB")
        save(self.d_B, "discriminatorB")
        save(self.g_BA, "generatorBA")
        save(self.d_A, "discriminatorA")

if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=300, batch_size=16, sample_interval=200)
