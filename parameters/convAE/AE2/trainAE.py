import tensorflow as tf
import numpy as np
from Model.convAE import convAE
import glob
import Pre_processing.GetInput as GetInput
import visualization.visual as visual
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2006_10_10\\scene2-camera1.vid'
image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2007_05_24\\scene3-camera1.vid'

channels = [32, 64, 128]
hiddens = [1024, 256]
W_shapes = [5, 5, 5]
strides = [4, 4, 2]
batch_size = 5
image_size = [480, 640]

model = convAE(channels, hiddens, W_shapes, strides, batch_size, image_size)

image_list = glob.glob(image_path + '\\*.jpeg')

continuous = True
lr = 1e-5
expoch = 1000000
optimizer = tf.train.AdamOptimizer(lr).minimize(model.loss)
All_loss = []
temp_loss = []
step = []

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
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

for i in range(expoch):
    index = np.random.randint(0, len(image_list), [batch_size])
    images = []
    for j in range(batch_size):
        images.append(GetInput.getimage(image_list[index[j]]))
    loss, _ = sess.run(fetches=[model.loss, optimizer],
                       feed_dict={model.input: images})
    # loss, KL, recon, log_sigma, test, _ = sess.run(
    #     fetches=[model.loss, model.KL_loss, model.recon_loss, model.log_sigma, model.test, optimizer],
    #     feed_dict={model.input: images, model.gaussian: gaussian})

    loss = loss / batch_size
    temp_loss.append(loss)
    # print('log_sigma and test: ', log_sigma, ', \n', test)
    print('Step %d|| loss: %8f' % (i, loss), '|| index: ', index)
    if i % 500 == 0 and i > 1:
        model.saver.save(sess, './Parameters/convAE/AE_', global_step=i)
        All_loss.append(np.average(temp_loss))
        step.append(i)
        visual.plot_AE_loss(All_loss, step)

        temp_loss = []
        temp_KL = []
        temp_recon = []


