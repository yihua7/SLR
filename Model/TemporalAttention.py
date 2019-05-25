import tensorflow as tf
import numpy as np
import json
import os
import Model.networks as networks
import Pre_processing.GetInput as GetInput
import visualization.visual as visual


class SelfAttention:
    def __init__(self, channel, conv_d, W_shape, fdim, hdim, classNum, lr):
        self.channel = channel
        self.conv_d = conv_d
        self.W_shape = W_shape
        self.fdim = fdim
        self.hdim = hdim
        self.classNum = classNum
        self.input = tf.placeholder(tf.float32, shape=[1, None, 64, 64, 3], name="input")
        self.label = tf.placeholder(tf.float32, shape=[None, self.classNum], name='label')
        self.output = self.predict(self.input)
        self.loss = tf.reduce_sum(tf.multiply(self.label, tf.divide(1, self.output+1e-7)))
        self.loss_square = tf.nn.l2_loss(tf.subtract(self.label, self.output))
        self.lr = lr
        self.optimimzer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=2)

    def predict(self, input):
        features = networks.set_cnn(input, self.conv_d, self.W_shape, self.channel, out_dim=self.fdim, scope='CNN')
        Q = self.Full(features, "Query")
        V = self.Full(features, "Value")
        K = tf.transpose(self.Full(features, "Keys"))
        Attention = tf.nn.softmax(tf.divide(tf.matmul(Q, K), np.sqrt(self.fdim)))
        Z = tf.matmul(Attention, V)
        value = tf.reduce_sum(Z, [0])
        value = tf.reshape(value, [1, -1])
        value = tf.contrib.layers.batch_norm(value, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
        self.value1 = value
        value = networks.set_full(value, self.hdim, scope='late_layer', activate=tf.nn.relu)
#        value = tf.contrib.layers.batch_norm(value, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
        self.value2 = value
        prediction = networks.set_full(value, self.classNum, scope='last_layer', activate=tf.nn.softmax)
        return prediction

    def Full(self, input, scope=None):
        with tf.variable_scope(scope or 'Full'):
            w = tf.get_variable('w', [self.fdim, self.hdim], dtype=tf.float32, initializer=tf.random_normal_initializer)
            return tf.matmul(input, w)

    def train(self, jsonPath, imagePath, expoch, coninue_train=False):
        # Get Data
        jsonfile = json.load(open(jsonPath))
        sess = tf.Session()
        if not coninue_train:
            sess.run(tf.global_variables_initializer())
        else:
            latest = tf.train.latest_checkpoint('./Parameters/Temporal_Attention/')
            self.saver.restore(sess, latest)
        plot_loss = []
        plot_step = []
        for i in range(expoch):
            temp_loss = 0
            count = 0
            for key in jsonfile:
                if os.path.exists(imagePath + jsonfile[key][0] + '\\spatial\\downSample\\'):
                    count += 1
                    label = np.zeros([1, self.classNum])
                    label[0, jsonfile[key][3]] = 1
                    data = []
                    start = int(jsonfile[key][1])
                    end = int(jsonfile[key][2])
                    for j in range(start, end + 1):
                        data .append(GetInput.getimage(imagePath + jsonfile[key][0] + '\\spatial\\downSample\\' + str(j) + '.jpeg'))

                    loss, output, _ = sess.run([self.loss, self.output, self.optimimzer],
                                               feed_dict={self.input: [data], self.label: label})
                    temp_loss += loss
                    intout = np.zeros(self.classNum)
                    intout[np.argmax(output)] = 1
                    outcome = str(list(np.squeeze(intout))==list(np.squeeze(label)))
                    print("Iteration: %d|    Loss: %.8f|     Outcome: " % (i, loss) + outcome + '|  ' + key + ' ' + jsonfile[key][0] + " " + str(list(output)))

            if i % 50 == 0 and i != 0:
                self.saver.save(sess, './Parameters/Temporal_Attention/temporal_', global_step=i)

            if i % 50 == 0 and i != 0:
                plot_loss.append(temp_loss/count)
                plot_step.append(len(plot_step))
                visual.plot_info(plot_loss, plot_step, name='_temporal')

    def test(self, jsonPath, imagePath):
        # Get Data
        jsonfile = json.load(open(jsonPath))
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        latest = tf.train.latest_checkpoint('./Parameters/Temporal_Attention')
        self.saver.restore(sess, latest)
#        sess.run(tf.global_variables_initializer())
        count = 0
        value1 = 0
        value2 = 0
        for key in jsonfile:
            if os.path.exists(imagePath + jsonfile[key][0] + '\\spatial\\downSample\\'):
                count += 1
                label = np.zeros([1, self.classNum])
                label[0, jsonfile[key][3]] = 1
                data = []
                start = int(jsonfile[key][1])
                end = int(jsonfile[key][2])
                for j in range(start, end + 1):
                    data .append(GetInput.getimage(imagePath + jsonfile[key][0] + '\\spatial\\downSample\\' + str(j) + '.jpeg'))

                loss, output = sess.run([self.loss, self.output],
                                        feed_dict={self.input: [data], self.label: label})
                if(count > 1):
                    [v1, v2] = [value1, value2]
                value1, value2 = sess.run([self.value1, self.value2],
                                          feed_dict={self.input: [data], self.label: label})
                #if(count > 1):
                #    print(np.linalg.norm(v1 - value1))
                #    print(np.linalg.norm(v2 - value2))

                intout = np.zeros(self.classNum)
                intout[np.argmax(output)] = 1
                outcome = str(list(np.squeeze(intout))==list(np.squeeze(label)))
                print(str(list(output)) + " Loss: %.8f|     Outcome: " % loss + outcome + '|  ' + key + '  ' + jsonfile[key][0])
                #print(str(list(value1)) + "\n" + str(list(value2)))

