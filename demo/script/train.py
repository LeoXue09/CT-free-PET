import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import h5py
import pandas as pd
import time
import os
import numpy as np 
import keras.backend as K
from keras.callbacks import TensorBoard
K.set_image_data_format("channels_first")

from Unet_3D_stride import unet_model_3d

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class SR_UnetGAN():
    def __init__(self):
        self.data_path = '/media/uni/ST_2/Attenuation/Data/h5_file/SH_Vision_ratio_downorder_4_112_prepad/data.h5'
        self.save_dir = '/media/uni/ST_2/Attenuation/result/SH_Vision_ratio_downorder_4_112_prepad_new_weight'
        # self.img_shape = (1,64,64,64)
        self.img_shape = (1,112,112,112)
        self.common_optimizer = Adam(0.0002, 0.5)
        self.epochs = 100
        self.batch_size = 1

        self.generator = self.build_generator()
        self.generator.compile(loss="mse", optimizer=self.common_optimizer)

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.common_optimizer)

        # Then, create and compile the adversarial model
        self.adversarial_model = self.build_adversarial_model()
        self.adversarial_model.compile(loss=['binary_crossentropy', 'mse'], 
                                        # loss_weights=[1e-1, 1e+3], 
                                        loss_weights=[1e-2, 1e+2], 
                                        optimizer=self.common_optimizer)

    def build_generator(self):
        model = unet_model_3d(self.img_shape)
        return model

    def build_discriminator(self):
        leakyrelu_alpha = 0.2
        momentum = 0.8

        input_shape = self.img_shape
        input_layer = Input(shape=input_shape)

        # Add the first convolution block
        dis1 = Conv3D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
        dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

        # Add the 2nd convolution block
        dis2 = Conv3D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
        dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
        dis2 = BatchNormalization(momentum=momentum)(dis2)

        # Add the third convolution block
        dis3 = Conv3D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
        dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
        dis3 = BatchNormalization(momentum=momentum)(dis3)

        # Add the fourth convolution block
        dis4 = Conv3D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
        dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
        dis4 = BatchNormalization(momentum=0.8)(dis4)

        # Add the fifth convolution block
        dis5 = Conv3D(256, kernel_size=3, strides=1, padding='same')(dis4)
        dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
        dis5 = BatchNormalization(momentum=momentum)(dis5)

        # Add the sixth convolution block
        dis6 = Conv3D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
        dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
        dis6 = BatchNormalization(momentum=momentum)(dis6)

        # Add the seventh convolution block
        dis7 = Conv3D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
        dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
        dis7 = BatchNormalization(momentum=momentum)(dis7)

        # Add the eight convolution block
        dis8 = Conv3D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
        dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
        dis8 = BatchNormalization(momentum=momentum)(dis8)

        # Add a dense layer
        dis8 = Flatten()(dis8)
        dis9 = Dense(units=1024)(dis8)
        dis9 = LeakyReLU(alpha=0.2)(dis9)

        # Last dense layer - for classification
        output = Dense(units=1, activation='sigmoid')(dis9)

        model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
        print(model.summary())
        return model

    def build_adversarial_model(self):

        input_A = Input(shape=self.img_shape)
        # input_B = Input(shape=self.img_shape)

        generated_B = self.generator(input_A)

        # Make the discriminator network as non-trainable
        self.discriminator.trainable = False

        # Get the probability of generated high-resolution images
        probs = self.discriminator(generated_B)

        # Create and compile an adversarial model
        model = Model([input_A], [probs, generated_B])
        return model

    def data_generator(self, data, name, length):
        while True:
            x_batch, y_batch = [], []
            count = 0
            for j in range(length):
                x = data.get(name+'_NAC_PET')[j]
                # y = data.get(name+'_AC_PET')[j]
                y = data.get(name+'_ratio')[j]
                x = np.expand_dims(x, axis=0)
                x_batch.append(x)
                y = np.expand_dims(y, axis=0)
                y_batch.append(y)
                count += 1
                if count == self.batch_size:
                    yield np.array(x_batch), np.array(y_batch)
                    count = 0
                    x_batch, y_batch = [], []

    def write_log(self, callback, name, value, batch_no):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

    def train(self):
        data = h5py.File(self.data_path, mode='r')
        train_filenames = np.array(data['train_filenames'])
        data_generator = self.data_generator(data, 'train', len(train_filenames))

        log_dir = os.path.join(self.save_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard.set_model(self.generator)
        tensorboard.set_model(self.discriminator)
        tensorboard.set_model(self.adversarial_model)
        batch_df = pd.DataFrame()
        epoch_df = pd.DataFrame()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            sess.run(tf.global_variables_initializer())
            K.set_session(sess)
            # Add a loop, which will run for a specified number of epochs:
            for epoch in range(1, self.epochs+1):
                # Create two lists to store losses
                gen_losses, dis_losses, adv_losses = [],[],[]
                number_of_batches = int(len(train_filenames) / self.batch_size)
                for index in range(number_of_batches):
                    print('Epoch: {}/{}\n  Batch: {}/{}'.format(epoch, self.epochs, index+1, number_of_batches))
                    x_batch, y_batch = next(data_generator)
                    gen_x = self.generator.predict(x_batch,verbose=3)
                    # Make the discriminator network trainable
                    self.discriminator.trainable = True
                               
                    # Create fake and real labels
                    labels_real = np.ones((self.batch_size ,1))
                    labels_fake = np.zeros((self.batch_size ,1))
                            
                    # Train the discriminator network
                    loss_real = self.discriminator.train_on_batch(y_batch, labels_real)
                    loss_fake = self.discriminator.train_on_batch(gen_x, labels_fake)
                            
                    # Calculate total discriminator loss
                    d_loss = 0.5 * (loss_real + loss_fake)

                    # Train the adversarial model
                    a_loss, prob_loss, g_loss = self.adversarial_model.train_on_batch([x_batch],
                                                                   [labels_real, y_batch])

                    gen_losses.append(g_loss)
                    dis_losses.append(d_loss)
                    adv_losses.append(a_loss)
                    print("    Pro_loss: {}\n    G_loss: {}\n    D_loss: {}\n    Adv_loss: {}".format(prob_loss,g_loss, d_loss, a_loss))

                batch_df = batch_df.append(pd.DataFrame({'epoch': [epoch] * len(gen_losses),
                                                         'batch': np.arange(1, len(gen_losses)+1),
                                                         'generator_loss': gen_losses, 
                                                         'discriminator_loss': dis_losses, 
                                                         'adversarial_loss': adv_losses}))
                epoch_df = epoch_df.append(pd.DataFrame({'epoch': [epoch],
                                                         'generator_loss': [np.mean(gen_losses)], 
                                                         'discriminator_loss': [np.mean(dis_losses)], 
                                                         'adversarial_loss': [np.mean(adv_losses)]}))
                batch_df = batch_df[['epoch', 'batch', 'generator_loss', 'discriminator_loss', 'adversarial_loss']]
                epoch_df = epoch_df[['epoch', 'generator_loss', 'discriminator_loss', 'adversarial_loss']]
                batch_df.to_csv(os.path.join(log_dir, 'batch_loss.csv'), index=False)
                epoch_df.to_csv(os.path.join(log_dir, 'epoch_loss.csv'), index=False)
                # Save losses to Tensorboard
                self.write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)
                self.write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
                self.write_log(tensorboard, 'adversarial_loss', np.mean(adv_losses), epoch)

                model_dir = os.path.join(self.save_dir, 'model')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                if self.epochs >= 40:
                    if epoch % 10 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                        self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
                else:
                    if epoch % 1 == 0:
                        self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
                        self.discriminator.save_weights(os.path.join(model_dir, 'discriminator_epoch_{}.hdf5'.format(epoch)))
        data.close()

if __name__ == '__main__':
    gan = SR_UnetGAN()
    gan.train()