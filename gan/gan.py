from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

from copy import copy

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        
        # build the model
        self.discriminator = self.discriminator_model()
        self.generator     = self.generator_model()
        
        self.discriminator.compile(loss = 'binary_crossentropy',
            optimizer = optimizer,
            metrics = ['accuracy'])
        self.discriminator.trainable = False
        
        self.gan = self.gan_model()
        self.gan.compile(loss = 'binary_crossentropy', 
            optimizer = optimizer,
            metrics = ['accuracy'])
        
    def generator_model(self):

        input_x = Input(shape=(self.latent_dim, ))
        x = Dense(256)(input_x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dense(np.prod(self.img_shape), activation = 'tanh')(x)
        x = Reshape(self.img_shape)(x)
        
        generator = Model(inputs = input_x, outputs = x)
        return generator

    def discriminator_model(self):

        input_x = Input(shape = self.img_shape)
        x = Reshape((np.prod(self.img_shape),))(input_x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Dense(1, activation = 'sigmoid')(x)
        
        discriminator = Model(inputs = input_x, outputs = x)
        return discriminator
    
    def gan_model(self):
        
        input_x = Input(shape = (self.latent_dim,))
        gene_x  = self.generator(input_x)
        y       = self.discriminator(gene_x)
        
        gan = Model(inputs = input_x, outputs = y)
        return gan
    
    def train(self, X, epochs = 1000, batch_size=128):
        disp_freq = 50
        save_freq = 50

        for epoch in range(epochs):

            # select a random batch of images
            idx = np.random.randint(0, X.shape[0], batch_size)
            
            # train the discriminator
            Xr = X[idx]
            yr = [1]*batch_size

            noise = np.random.normal(size = (batch_size, self.latent_dim))

            Xf = self.generator.predict(noise)
            yf = [0]*batch_size
            
            d_loss_real = self.discriminator.train_on_batch(Xr, yr)
            d_loss_fake = self.discriminator.train_on_batch(Xf, yf)
            d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
            
            # train the generator
            noise = np.random.normal(size = (batch_size, self.latent_dim))
            yr = [1]*batch_size
            
            # train the generator 
            g_loss = self.gan.train_on_batch(noise, yr)


            if epoch % disp_freq == 0:
                print ("epoch %d D acc.: %.2f G acc.: %.2f" % (epoch, d_loss[1], g_loss[1]))

            if epoch % save_freq == 0:
                self.sample_images(epoch)
                
                self.discriminator.save('result_model/discriminator' + str(epoch) + '.h5')
                self.generator.save('result_model/generator' + str(epoch) + '.h5')
                self.gan.save('result_model/gan' + str(epoch) + '.h5')
                
        return None

    def sample_images(self, epoch):
        noise = np.random.normal(size = (25, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # rescale images 
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('result_image/' + str(epoch) + '.png')
        plt.close()
        
        return None

if __name__ == '__main__':
    # load the dataset
    X, _ = mnist.load_data()[0]
    X = X / 127.5 - 1.
    X = np.expand_dims(X, axis=3)
    
    # model
    gan = GAN()
    gan.train(X, epochs=30000, batch_size=32)
