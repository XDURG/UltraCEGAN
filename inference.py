from __future__ import print_function, division
import os
import numpy as np
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from metrics import ssim_metric, psnr_metric, mse_metric, mae_metric
from data_loader import DataLoader

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
    d2 = conv2d(d1, self.gf * 2)
    d3 = conv2d(d2, self.gf * 4)
    d4 = conv2d(d3, self.gf * 8)
    d5 = conv2d(d4, self.gf * 16)
    u1_0 = deconv2d(d5, d4, self.gf * 8)
    u1 = deconv2d(u1_0, d3, self.gf * 4)
    u2 = deconv2d(u1, d2, self.gf * 2)
    u3 = deconv2d(u2, d1, self.gf)
    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
    return Model(d0, output_img)

def test_main():
    data_loader = DataLoader(dataset_name='sythetic_ceus', img_res=(256, 256))
    img_shape = (256, 256, 3)
    model = build_generator()
    output_path = './output_path'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_path = './sythetic_ceus_250_600_weights.hdf5'
    model.load_weights(model_path)
    path_valA = "../valA/"
    path_valB = "../valB/"
    pathDir_img = [i for i in os.listdir(path_valA)]
    MSE = []
    MAE = []
    SSIM =[]
    PSNR =[]
    for i in range(len(pathDir_img)):
        imgA = data_loader.load_img(path_valA + pathDir_img[i])
        imgB = data_loader.load_img(path_valB + pathDir_img[i])
        predictA = model.predict(imgA)
        predictA = np.array(predictA[0])
        imgB = np.array(imgB[0])
        imgB = 0.5 * imgB + 0.5
        MSE.append(mse_metric(predictA*255, imgB*255))
        MAE.append(mae_metric(predictA*255, imgB*255))
        SSIM.append(ssim_metric(predictA*255, imgB*255))
        PSNR.append(psnr_metric(predictA*255, imgB*255))
        print(i, np.sum(MSE) / i, np.sum(MAE) / i, np.sum(SSIM) / i, np.sum(PSNR) / i)

if __name__ == '__main__':
    test_main()


