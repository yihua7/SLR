import tensorflow as tf
import numpy as np
from Model.convAE_new import convAE_test
import glob
import Pre_processing.GetInput as GetInput
import visualization.visual as visual
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

main_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG'

channels = [10, 20, 20, 20, 10]
hiddens = [1024]
W_shapes = [3, 3, 3, 3, 3]
strides = [2, 2, 2, 2, 2]
batch_size = 5
image_size = [128, 128]

model = convAE_test(channels, hiddens, W_shapes, strides, batch_size, image_size)

scene_list = []
data_list = glob.glob(main_path + '/ASL*')
for i in data_list:
    scene_list += (glob.glob(i + '\\*vid'))

continuous = False
lr = 1e-3
learning_rate = tf.placeholder(tf.float32, [])
expoch = 1000000
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
All_loss = []
temp_loss = []
step = []

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config=config)

if continuous:
    sess.run(tf.global_variables_initializer())
    latest = tf.train.latest_checkpoint('./Parameters/convAE/')
    model.saver.restore(sess, latest)
else:
    sess.run(tf.global_variables_initializer())


count = 0
for i in range(expoch):
    index_scene = np.random.randint(0, len(scene_list), [batch_size])
    images = []
    for j in range(batch_size):
        images_list = glob.glob(scene_list[index_scene[j]] + '/*.jpeg')
        index_image = np.random.randint(0, len(images_list))
        images.append(GetInput.getimage(images_list[index_image]))
    loss, embedded, _ = sess.run(fetches=[model.loss, model.embedded, optimizer],
                                 feed_dict={model.raw_input: images, learning_rate: lr})
    embedded = np.array(embedded)
    d = (embedded - np.average(embedded, 0))
    d = d*d
    d = np.average(d)

    loss = loss / batch_size
    temp_loss.append(loss)
    print('Step %d|| loss: %8f' % (i, loss), ' || covariance: ', d)
    if i % 200 == 0 and i > 1:
        model.saver.save(sess, './Parameters/convAE/AE_', global_step=i)
        All_loss.append(np.average(temp_loss))
        step.append(i)
        visual.plot_AE_loss(All_loss, step)
        if All_loss[0] > All_loss[-1] * 10 or len(step) > 50:
            if len(step) > 50:
                lr = 1e-5
            count += 1
            visual.plot_AE_loss(All_loss, step, 'AE_' + str(count) + '.png')
            All_loss = []
            step = []

        temp_loss = []
        temp_KL = []
        temp_recon = []


