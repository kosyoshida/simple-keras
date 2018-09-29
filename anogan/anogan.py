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

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from copy import copy

import os
os.chdir('../')
from dcgan import dcgan
os.chdir('anogan')

def feature_extractor_model(discriminator):
    discriminator.load_weights('result_weight/discriminator.h5') 
    feature_extractor = Model(inputs=discriminator.layers[0].input, outputs=discriminator.layers[-2].output)
    feature_extractor.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return feature_extractor

def view_feature(feature_extractor, X_norm, X_abnorm):
    f_norm   = feature_extractor.predict(X_norm)
    f_abnorm = feature_extractor.predict(X_abnorm)
    f = np.concatenate((f_norm, f_abnorm), axis = 0)
    
    n_normal   = len(f_norm)
    
    f_emb = TSNE(n_components = 2).fit_transform(f)
    plt.figure()
    plt.scatter(f_emb[:n_normal, 0], f_emb[:n_normal,1], 
                s = 0.2, c = 'b', label = 'normal (=6)')
    plt.scatter(f_emb[n_normal:, 0], f_emb[n_normal:,1], 
                s = 0.2, c = 'r', label = 'normal (=8)')
    plt.title('t-SNE embedding of feature')
    plt.legend()
    plt.savefig('t-SNE_embedding.png')
    plt.show()    
    return None

if __name__ == '__main__':
    # load the dataset
    X, y = mnist.load_data()[0]
    X = X / 127.5 - 1.
    X = np.expand_dims(X, axis=3)
    
    # normal => y = 6
    idx_norm = np.where(y == 6)
    X_norm   = X[idx_norm]
    X_norm_trn, X_norm_tst = train_test_split(X_norm, test_size = 0.1)
    
    # abnormal => y = 8
    idx_abnorm = np.where(y == 8)
    X_abnorm   = X[idx_abnorm]
    _, X_abnorm_tst = train_test_split(X_abnorm, test_size = 0.1)
    
    # model
    gan = dcgan.GAN()
    gan.train(X_norm_trn, epochs=500, batch_size=32)
    
    gan.generator.save_weights('result_weight/generator.h5')
    gan.discriminator.save_weights('result_weight/discriminator.h5')
    
    feature_extractor = feature_extractor_model(gan.discriminator)
    view_feature(feature_extractor, X_norm_tst, X_abnorm_tst)
