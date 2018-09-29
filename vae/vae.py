from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

from copy import copy

class VAE():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        optimizer = Adam(0.0002, 0.5)
        
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        
        self.vae = self.vae_model()
        self.vae.compile(optimizer = optimizer,
                         loss = self.vae_loss)


    # encoder
    def encoder_model(self):
        input_x = Input(shape = (784, ))
        x = Dense(256, activation = 'relu')(input_x)
        x = Dense(32, activation = 'relu')(x)
        
        z_mean = Dense(2, name='z_mean')(x)
        z_log_var = Dense(2, name='z_log_var')(x)
        
        encoder = Model(inputs = input_x, outputs = [z_mean, z_log_var])
        return encoder

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # decoder
    def decoder_model(self):
        input_x = Input(shape = (2,))
        x = Dense(32, activation = 'relu')(input_x)
        x = Dense(256, activation = 'relu')(x)
        output_x = Dense(784, activation = 'sigmoid')(x)
        
        decoder = Model(inputs = input_x, outputs = output_x)
        return decoder
    
    # vae
    def vae_model(self):
        input_x = Input(shape = (784, ))
        z_mean, z_log_var = self.encoder(input_x)
        z = Lambda(self.sampling)([z_mean, z_log_var])
        output_x = self.decoder(z)
        K.clip(output_x, 1e-8, 1 - 1e-8)
        
        vae = Model(inputs = input_x, outputs = output_x)
        
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        
        return vae
    
    def vae_loss(self, input_x, output_x):
        loglikelihood = -K.sum(K.binary_crossentropy(input_x, output_x), axis = -1)
        kl_divergence = 0.5*K.sum(K.square(self.z_mean) + K.square(self.z_log_var) - K.log(1e-8 + K.square(self.z_log_var)) -1, axis=-1)
        return -loglikelihood + kl_divergence
  
    def train(self, X, y, epochs = 1000, batch_size=128):
        disp_freq = 50
        save_freq = 200
        for epoch in range(epochs):

            # select a random batch of images
            idx = np.random.randint(0, X.shape[0], batch_size)
            
            # train the discriminator
            X_ = X[idx]
            
            # train the vae 
            vae_loss = self.vae.train_on_batch(X_, X_)

            if epoch % disp_freq == 0:
                print ("epoch %d loss.: %.2f" % (epoch, vae_loss))
            if epoch % save_freq == 0:
                self.sample_image(X, y, epoch)
                
                self.encoder.save('result_model/encoder' + str(epoch) + '.h5')
                self.vae.save('result_model/vae' + str(epoch) + '.h5')
        return None
    
    def sample_image(self, X, y, epoch):
        z_mean, _ = self.encoder.predict(X)
        
        plt.figure()
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c = y, marker = 'o', s = 0.2, cmap = 'jet')
        plt.colorbar(ticks=range(10))
        plt.grid(True)
        plt.savefig('result_image/' + str(epoch) + '.png')
        plt.close()
        
        return None
        
if __name__ == '__main__':
    # load the dataset
    X, y = mnist.load_data()[0]
    X = X / 255.
    X = np.reshape(X, (-1, 784))
    
    # model
    vae = VAE()
    vae.train(X, y, epochs=10000, batch_size=32)
