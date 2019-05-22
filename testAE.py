import tensorflow as tf
import numpy as np
from Model.convAE import convAE
import glob
import Pre_processing.GetInput as GetInput
import visualization.visual as visual
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2006_10_10\\scene2-camera1.vid'
image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2007_05_24\\scene3-camera1.vid'
# image_path = 'C:\\Users\\agogow5\\Desktop\\Test'
# image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2006_10_10\\scene2-camera1.vid\\spatial\\downSample'

channels = [10, 10, 10, 10]
hiddens = [128]
W_shapes = [3, 3, 3, 3]
strides = [2, 2, 2, 2]
batch_size = 5
# image_size = [64, 64]
image_size = [256, 256]

model = convAE(channels, hiddens, W_shapes, strides, 1, image_size)

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
latest = tf.train.latest_checkpoint('./parameters/convAE/')
model.saver.restore(sess, latest)
total_embedded = []

for k in image_list:
    images = [(GetInput.getimage(k))]
    recon = sess.run(fetches=[model.recon],
                     feed_dict={model.input: images})

    print(k)
    [embedded] = sess.run(fetches=[model.embedded], feed_dict={model.raw_input: images})
    total_embedded.append(embedded)
    recon = np.squeeze(recon)
    visual.save_image(recon, image_path + '\\AErecon\\', k.split('\\')[-1].split('.')[0])

visual.plot_embedded(total_embedded, image_path + '\\AErecon\\Embedding.png')
