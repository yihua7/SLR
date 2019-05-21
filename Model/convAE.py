import tensorflow as tf
import numpy as np
import Model.networks as nw
import math


class convAE:
    def __init__(self, channels, hiddens, W_shapes, strides, batch_size, image_size):
        self.channels = channels
        self.hiddens = hiddens
        self.W_shapes = W_shapes
        self.strides = strides
        self.batch_size = batch_size
        self.input = tf.placeholder(tf.float32, shape=[None, image_size[0], image_size[1], 3], name='Image')
        final_frame = self.input.get_shape().as_list()[1:]
        final_frame[0] = math.ceil(final_frame[0] / np.cumprod(strides)[-1])
        final_frame[1] = math.ceil(final_frame[1] / np.cumprod(strides)[-1])
        final_frame[2] = self.channels[-1]
        self.final_frame = final_frame
        self.ff_dim = final_frame[0] * final_frame[1] * final_frame[2]

        #  Calculate Reconstruction Loss
        self.embedded = self.encoder(self.input)
        self.recon = self.decoder(self.embedded)
        self.recon_loss = tf.nn.l2_loss(tf.subtract(self.input, self.recon))
        print('Recon Loss: ', self.recon_loss)

        # Total Loss
        self.loss = self.recon_loss
        print('Loss: ', self.loss)

        self.saver = tf.train.Saver(max_to_keep=2)

    def encoder(self, input):
        with tf.variable_scope('Encoder'):
            print('Encoder')
            for i in range(len(self.channels)):
                input = nw.set_conv(input, self.W_shapes[i], self.channels[i], self.strides[i], 'conv' + str(i), bn=False)
                print(input)
            input = tf.reshape(input, [-1, self.ff_dim])
            print(input)
            for i in range(len(self.hiddens) - 1):
                input = nw.set_full(input, self.hiddens[i], 'full' + str(i), bn=False)
                print(input)
            input = tf.contrib.layers.batch_norm(input, .9, epsilon=1e-5, activation_fn=None)
            output = nw.set_full(input, self.hiddens[-1], 'mean', None)
            print(output)
            return output

    def decoder(self, input):
        print('Decoder')
        sample = input
        print(sample)
        with tf.variable_scope('Decoder'):
            for i in range(len(self.hiddens) - 1):
                sample = nw.set_full(sample, self.hiddens[len(self.hiddens) - i - 2], 'full' + str(i), bn=False)
                print(sample)
            sample = nw.set_full(sample, self.ff_dim, 'full_last', bn=False)
            print(sample)
            sample = tf.reshape(sample, np.concatenate([[-1], self.final_frame], 0))
            print(sample)
            for i in range(1, len(self.channels)):
                index = len(self.channels) - i - 1
                sample = nw.set_deconv(sample, self.W_shapes[index], self.channels[index], self.strides[index+1], self.batch_size, 'deconv' + str(i), bn=False)
                print(sample)
            sample = nw.set_deconv(sample, self.W_shapes[0], 3, self.strides[0], self.batch_size, 'last_deconv', tf.nn.sigmoid, bn=True)
            sample = tf.multiply(sample, 255.)
            print(sample)
            print('End Decoder')
            return sample

