from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Activation
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import numpy as np
import matplotlib.pylab as plt
import imutils

from copy import copy

class BNN():

    def __init__(self, n_input, n_hidden, n_output, dropout, lmbd):
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dropout  = dropout
        self.lmbd     = lmbd

        optimizer = Adam(0.0002, 0.5)

        self.model = self.cls_model()

        self.model.compile(optimizer = optimizer,
                           loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def cls_model(self):
        input_x = Input(shape = [self.n_input, ])
        h = Dropout(self.dropout)(input_x)
        h = Dense(self.n_hidden, kernel_regularizer = l2(self.lmbd), 
                  bias_regularizer = l2(self.lmbd))(h)
        h = Activation(activation = 'relu')(h)
        h = Dropout(self.dropout)(h)
        h = Dense(self.n_output, kernel_regularizer = l2(self.lmbd),
                  bias_regularizer = l2(self.lmbd))(h)
        output_x = Activation(activation = 'softmax')(h)

        model =  Model(inputs = input_x, outputs = output_x)
        return model

    def train(self, X, y, batch_size, epochs):
        self.model.fit(X, y, batch_size, epochs, validation_split = 0.1)
        return self.model

    def get_uncertainty(self, X, T = 100):
        def myrelu(x):
            return np.maximum(np.zeros_like(x), x)
        def mysoftmax(x):
            return np.exp(x) / np.sum(np.exp(x))

        w1, b1, w2, b2 = self.model.get_weights()
        output = []
        for t in range(T):
            z1 = np.diag(np.random.rand(self.n_input) > self.dropout).astype('float32')
            z2 = np.diag(np.random.rand(self.n_hidden) > self.dropout).astype('float32')
            h = np.dot(X, z1).dot(w1) + b1
            h = myrelu(h)
            h = np.dot(h, z2).dot(w2) + b2
            h = mysoftmax(h)
            output.append(h)
        output = np.array(output)
        return output

    def sample_image(self, x_img):
        rots = [0, 60, 120, 180]
        fig, axs = plt.subplots(len(rots), 2, figsize = (6,12))
        for i, rot in enumerate(rots):
            x_img_noise = imutils.rotate(x_img, rot).reshape([784,])
            output = self.get_uncertainty(x_img_noise)
            x_img_noise = np.reshape(x_img_noise, [28,28])

            axs[i][0].imshow(x_img_noise, cmap = 'gray')
            axs[i][1].boxplot(output)
            if i == 0:
                axs[i][0].set_title('rotated data', fontsize = 16)
                axs[i][1].set_title('uncertainty of softmax', fontsize = 16)
            axs[i][0].set_xticklabels([])
            axs[i][0].set_yticklabels([])
            axs[i][0].set_xlabel('rotation {} [deg]'.format(rot))
            axs[i][1].set_ylim([0, 0.6])
            axs[i][1].set_xticklabels(np.arange(10))  
        plt.savefig('result_image/result.png')

if __name__ == '__main__':
    # load the dataset
    (X_trn, y_trn), (X_tst, _) = mnist.load_data()
    X_trn = X_trn.reshape([-1, 28*28]).astype('float32') / 255.
    X_tst = X_tst.reshape([-1, 28*28]).astype('float32') / 255.
    y_trn = to_categorical(y_trn, num_classes = 10)

    # parameters
    n_input  = 784
    n_hidden = 128
    n_output = 10
    dropout = 0.3
    lmbd = 0.05

    # model training
    bnn = BNN(n_input, n_hidden, n_output, dropout, lmbd)
    bnn.train(X_trn, y_trn, 256, 10)

    # test
    x_img = X_tst[0].reshape([28,28])
    bnn.sample_image(x_img)
