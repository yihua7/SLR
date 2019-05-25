import tensorflow as tf
import numpy as np
from Model.convVAE import convVAE
import glob
import Pre_processing.GetInput as GetInput
import visualization.visual as visual
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2006_10_10\\scene2-camera1.vid'

channels = [16, 64, 128]
hiddens = [128, 64]
W_shapes = [3, 3, 3]
strides = [2, 2, 2]
batch_size = 5

model = convVAE(channels, hiddens, W_shapes, strides, 1., 1)

image_list = glob.glob(image_path + '\\*.jpeg')

continuous = False
lr = .01
expoch = 1000000
optimizer = tf.train.AdamOptimizer(lr).minimize(model.loss)
All_loss = []
All_KL = []
All_recon = []
step = []

sess = tf.Session()
latest = tf.train.latest_checkpoint('./Parameters/convVAE/')
model.saver.restore(sess, latest)

for k in image_list:
    images = [(GetInput.getimage(k))]
    gaussian = np.random.normal(size=[1, 1])
    recon = sess.run(fetches=[model.recon],
                     feed_dict={model.input: images, model.gaussian: gaussian})

    print(k)
    recon = np.squeeze(recon)
    visual.save_image(recon, image_path + '\\recon\\', k.split('\\')[-1])
