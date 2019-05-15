import tensorflow as tf
import numpy as np
from Model.convVAE import convVAE
import glob
import Pre_processing.GetInput as GetInput
import visualization.visual as visual
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2006_10_10\\scene2-camera1.vid'

channels = [32, 128]
hiddens = [512, 128]
W_shapes = [3, 3, 3]
strides = [2, 2, 2]
batch_size = 5

model = convVAE(channels, hiddens, W_shapes, strides, 1., batch_size)

image_list = glob.glob(image_path + '\\*.jpeg')

continuous = False
lr = .01
expoch = 1000000
optimizer = tf.train.AdamOptimizer(lr).minimize(model.loss)
All_loss = []
All_KL = []
All_recon = []
step = []

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
if continuous:
    latest = tf.train.latest_checkpoint('./parameters/convVAE/')
    model.saver.restore(sess, latest)
else:
    sess.run(tf.global_variables_initializer())

for i in range(expoch):
    index = np.random.randint(0, len(image_list), [batch_size])
    images = []
    for j in range(batch_size):
        images.append(GetInput.getimage(image_list[index[j]]))
    gaussian = np.random.normal(size=[batch_size, 1])
    loss, KL, recon, _ = sess.run(fetches=[model.loss, model.KL_loss, model.recon_loss, optimizer],
                                  feed_dict={model.input: images, model.gaussian: gaussian})
    # loss, KL, recon, log_sigma, test, _ = sess.run(
    #     fetches=[model.loss, model.KL_loss, model.recon_loss, model.log_sigma, model.test, optimizer],
    #     feed_dict={model.input: images, model.gaussian: gaussian})

    loss = np.average(loss)
    KL = np.average(KL)
    recon = recon / batch_size
    # print('log_sigma and test: ', log_sigma, ', \n', test)
    print('Step %d|| loss: %8f || KL: %8f || recon: %8f || KL/recon: %2f' % (i, loss, KL, recon, KL/recon))
    if i % 50 == 0:
        model.saver.save(sess, './parameters/convVAE/VAE_', global_step=i)
        All_loss.append(loss)
        All_KL.append(KL)
        All_recon.append(recon)
        step.append(i)
        visual.plot_loss(All_loss, All_KL, All_recon, step)


