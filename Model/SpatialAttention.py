import tensorflow as tf
import numpy as np
import Model.networks as networks
import visualization.visual as visual
import Pre_processing.GetInput as GetInput
import os
import glob

project_path = ''


class Spatial_hourglass():
    def __init__(self, block_number, layers, out_dim, point_num, lr, training=True, dropout_rate=0.2):
        self.block_number = block_number
        self.layers = layers
        self.out_dim = out_dim
        self.point_num = point_num
        self.lr = lr
        self.training = training
        self.dropout_rate = dropout_rate
        self.rawinput = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        self.image = tf.image.resize_images(self.rawinput, [256, 256], method=2)
        self.rawlabel = tf.placeholder(tf.float32, shape=[None, None, None, point_num], name='input_label')
        self.label = tf.image.resize_images(self.rawlabel, [64, 64], method=2)
        self.alignimage = tf.image.resize_images(self.image, [64, 64], method=2)

        with tf.variable_scope('hourglass0_down_sampling'):
            self.mid = networks.set_conv(self.image, 6, 64, 2, 'compression')  # down sampling
            self.mid = networks.set_res(self.mid, 128, 'compression_res0')
            self.mid = tf.nn.max_pool(self.mid, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # down sampling
            self.mid = networks.set_res(self.mid, 128, 'compression_res1')
            self.mid = networks.set_res(self.mid, out_dim, 'compression_res2')

        hgout0 = networks.set_hourglass(input=self.mid, layers=layers, out_dim=out_dim, scope='hourglass0')
        with tf.variable_scope('hourglass0_back'):
            hgout_conv1 = networks.set_conv(hgout0, 1, out_dim, 1, 'hgout0_conv0')
            hgout_conv2 = networks.set_conv(hgout_conv1, 1, out_dim, 1, 'hgout0_conv1')

            pred = networks.set_conv(hgout_conv1, 1, point_num, 1, 'pred0')
            heat_map = [pred]
            heat_map_reshape = networks.set_conv(pred, 1, out_dim, 1, 'reshape0')

            hgin1 = tf.add_n([self.mid, hgout_conv2, heat_map_reshape])
        hgin = [hgin1]

        for i in range(1, self.block_number):
            hgout0 = networks.set_hourglass(input=hgin[i-1], layers=layers, out_dim=out_dim, scope='hourglass'+str(i))
            with tf.variable_scope('hourglass'+str(i)+'_back'):
                hgout0 = tf.layers.dropout(hgout0, rate=self.dropout_rate, training=self.training, name='dropout'+str(i))
                hgout_conv1 = networks.set_conv(hgout0, 1, out_dim, 1, 'hgout'+str(i)+'_conv0')
                hgout_conv1 = tf.contrib.layers.batch_norm(hgout_conv1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                           scope='hgout'+str(i)+'_batch')
                hgout_conv2 = networks.set_conv(hgout_conv1, 1, out_dim, 1, 'hgout'+str(i)+'_conv1')

                pred = networks.set_conv(hgout_conv2, 1, point_num, 1, 'pred'+str(i), activate="sigmoid")
                heat_map.append(pred)
                heat_map_reshape = networks.set_conv(pred, 1, out_dim, 1, 'reshape'+str(i))

                hgin1 = tf.add_n([hgin[i-1], hgout_conv2, heat_map_reshape])
            hgin.append(hgin1)

        # Customize your output
        # Mean Output
        self.output_mean = tf.reduce_mean(heat_map, 0)
        # Output on each step
        self.step_output = heat_map

        self.output = pred
        self.alignoutput = tf.image.resize_images(self.output, [480, 640], method=2)
        self.loss_sum = tf.nn.l2_loss(tf.subtract(heat_map, self.label))
        self.loss = tf.nn.l2_loss(tf.subtract(self.output, self.label))

        self.optimizer = tf.train.RMSPropOptimizer(lr).minimize(self.loss)
        self.optimizer_all = tf.train.RMSPropOptimizer(lr).minimize(self.loss_sum)

        var = []
        step_loss = []
        for i in range(self.block_number):
            step_loss.append(tf.nn.l2_loss(tf.subtract(heat_map[i], self.label)))
            var.append([v for v in tf.trainable_variables() if v.name.startswith("hourglass" + str(i))])
        self.step_loss = step_loss
        self.step_var = var

        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self, data_path, label_path, batch_size, maxepoch, continue_train=False, base=0, step='all'):

        if step == 'all':
            # Output is from last layer
            optimizer = self.optimizer_all
            loss_tensor = tf.reduce_sum(self.step_loss)
            output_tensor_mean = tf.reduce_mean(self.step_output, 0)
            output_tensor_last = self.output
        else:
            # Output is from step layer
            loss_tensor = tf.reduce_sum(tf.gather(self.step_loss, step))
            optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(loss_tensor)
            output_tensor_mean = tf.reduce_mean(tf.gather(self.step_output, step), 0)
            output_tensor_last = self.step_output[max(step)]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            tf.global_variables_initializer().run()
            if continue_train:
                latest = tf.train.latest_checkpoint('./Parameters/Spatial_Attention/spatial_')
                self.saver.restore(sess, latest)

            plot_loss = []
            plot_step = []

            # Feed batch to train
            labelTime = os.listdir(label_path)
            dataset_num = len(labelTime)
            time_index = 0

            dataset = labelTime[time_index]
            label_list = os.listdir(label_path + dataset + '/')
            data_list = os.listdir(data_path + dataset + '/')

            base = base if continue_train else 0
            labelfile_index = 0
            labelfile = label_list[labelfile_index]
            datafile = [f for f in data_list if f.startswith(labelfile[0:-4])][0]
            labeldata = GetInput.getlabel(label_path+'\\'+dataset+'\\'+labelfile)
            num_img = np.shape(os.listdir(data_path+'\\'+dataset+'//'+datafile))[0]
            num_data = np.shape(labeldata)[0]
            start_data = 0
            for i in range(base, maxepoch):

                image = []
                if start_data > num_data - batch_size or start_data > num_img - batch_size:
                    if labelfile_index == np.shape(label_list)[0] - 1:
                        labelfile_index = 0
                        time_index = 0 if(time_index == dataset_num-1) else time_index+1
                        dataset_num = len(labelTime)
                        dataset = labelTime[time_index]
                        label_list = os.listdir(label_path + '/' + dataset + '/')
                        data_list = os.listdir(data_path + '/' + dataset + '/')
                    else:
                        labelfile_index += 1
                    labelfile = label_list[labelfile_index]
                    print("Move to label file: " + dataset + '/' + labelfile)
                    datafile = [f for f in data_list if f.startswith(labelfile[0:-4])][0]
                    labeldata = GetInput.getlabel(label_path+'\\'+dataset+'\\'+labelfile)
                    num_img = np.shape(os.listdir(data_path + '\\' + dataset + '//' + datafile))[0]
                    num_data = np.shape(labeldata)[0]
                    start_data = 0

                label, start_data, batch = GetInput.getlabel_batch(labeldata, batch_size, start_data)
                # Load Image and Label
                for j in batch:
                    # Load Image
                    image.append(GetInput.getimage(data_path+'\\'+dataset+'\\'+datafile+'\\'+str(j)+'.jpeg'))
                heatmap = GetInput.getheatmap(label, [np.shape(image)[1], np.shape(image)[2], 3])

                # Get Loss and Output(Prediction)
                loss, output, output_last = sess.run([loss_tensor, output_tensor_mean, output_tensor_last],
                                                     feed_dict={self.rawinput: image, self.rawlabel: heatmap})

                if i % 100 == 0:
                    alignimage, labelimage = sess.run([self.alignimage, self.label],
                                                      feed_dict={self.rawinput: [image[0]], self.rawlabel: [heatmap[0]]})
                    visual.hotmap_visualization(output[0], alignimage, labelimage, ".\\visualization\\Visual_Image\\Step\\"
                                                , str(i) + '.png')

                # Optimization
                sess.run([optimizer], feed_dict={self.rawinput: image, self.rawlabel: heatmap})

                print("Iteration: %5d | loss: %.8f " % (i, loss) + dataset + '/' + labelfile)

                # Print Training Information
                if i % 100 == 0 and i != 0:
                    plot_loss.append(loss)
                    plot_step.append(len(plot_step))
                    visual.plot_info(plot_loss, plot_step, name=str(step))
                    self.saver.save(sess, './Parameters/Spatial_Attention/spatial_', global_step=i)
        sess.close()

    def test(self, data_path, mode='downSample'):

        if mode == 'downSample':
            output_tensor_mean = tf.reduce_mean(self.step_output, 0)
            output = output_tensor_mean
            output_image = self.alignimage
        else:
            # Output is from last layer
            output = self.alignoutput
            output_image = self.rawinput

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        data_list = glob.glob(data_path + '/*.jpeg')

        with tf.Session(config=config) as sess:

            tf.global_variables_initializer().run()
            latest = tf.train.latest_checkpoint('./Parameters/Spatial_Attention')
            self.saver.restore(sess, latest)

            for name in data_list:
                print("Processing data: " + name)
                image = [GetInput.getimage(name)]

                # Get Loss and Output(Prediction)
                alignoutput = sess.run(output, feed_dict={self.rawinput: image})

                alignimage = sess.run([output_image],
                                      feed_dict={self.rawinput: image})
                visual.spatial_output(alignoutput[0], alignimage, data_path + '\\spatial\\' + mode + '\\'
                                      , name)
        sess.close()
