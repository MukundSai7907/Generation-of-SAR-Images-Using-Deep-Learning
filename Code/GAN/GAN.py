from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LayerNormalization
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
import os
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image
import numpy as np
import tensorflow as tf

##########################################################################################
class DCGAN():
    def __init__(self):

        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 32
        self.weight_decay = 1e-4

        optimizer = Adam(0.0001 , 0.5)
        optimizer1 = Adam(0.0001 , 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer1)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(4*4*128 , activation = 'linear' , input_dim = self.latent_dim))
        model.add(Reshape((4,4,128)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(UpSampling2D()) 
        model.add(Conv2DTranspose(filters = 64 , kernel_size = 5 , padding = 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(filters = 32 , kernel_size = 5 , padding = 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(UpSampling2D()) 
        model.add(Conv2DTranspose(filters = 16 , kernel_size = 5 , padding = 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(UpSampling2D()) 
        model.add(Conv2DTranspose(filters = 8 , kernel_size = 5 , padding = 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(filters = 8 , kernel_size = 5 , padding = 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2)) 
        model.add(Conv2D(filters = 4 , kernel_size = 5 , padding = 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2)) 
        model.add(Conv2D(filters = 1 , kernel_size = 5 , padding = 'same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2)) 

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=5, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization()) #
        model.add(LeakyReLU(alpha = 0.2))
        model.add(MaxPool2D())
        model.add(Conv2D(32, kernel_size=5, padding="same"))
        model.add(BatchNormalization()) #
        model.add(LeakyReLU(alpha = 0.2))
        model.add(MaxPool2D())
        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization()) #
        model.add(LeakyReLU(alpha = 0.2))
        model.add(MaxPool2D())
        model.add(Conv2D(128, kernel_size=5, padding="same"))
        model.add(BatchNormalization()) #
        model.add(LeakyReLU(alpha = 0.2))
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(1 , activation = 'sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)



    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        dirname = 'Path'
        X_train = []
        for filename in os.listdir(dirname):
          img = Image.open(dirname + filename)
          img = np.asarray(img)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          X_train.append(img)
      

        for x in X_train:
          x = x / 127.5 - 1
 
        X_train = np.expand_dims(X_train, axis=3)


        X_train = np.asarray(X_train)

        valid = []
        valid = [0.9 for i in range(batch_size)]
        valid = np.expand_dims(valid , axis=1)
 
        fake = []
        fake = [0.1 for i in range(batch_size)]
        fake = np.expand_dims(fake , axis=1)

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch(noise, valid)


            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 50*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_imgs(epoch)
            if epoch % 20 == 0:
                self.generator.save('Dirname' + str(epoch) + '.h5')

    def save_imgs(self, epoch):
        noise = np.random.normal(0, 1, (5, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5


        cnt = 0 
        for img in gen_imgs:
          img = np.squeeze(img , axis=2)
          plt.imsave("Dirname" + str(epoch) + '_' + str(cnt) + ".png" , img , cmap='gray')
          cnt+=1


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=2000, batch_size=8, save_interval=10)