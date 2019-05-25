import tensorflow as tf
import numpy as np
from Model.convAE_new import convAE_test
import glob
import Pre_processing.GetInput as GetInput
import visualization.visual as visual
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2006_10_10\\scene2-camera1.vid'
# image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2007_05_24\\scene3-camera1.vid'
# image_path = 'C:\\Users\\agogow5\\Desktop\\test'
# image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2006_10_10\\scene2-camera1.vid\\spatial\\downSample'
image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2008_05_29b\\scene9-camera1.vid'

channels = [10, 20, 20, 20, 10]
hiddens = [1024]
W_shapes = [3, 3, 3, 3, 3]
strides = [2, 2, 2, 2, 2]
batch_size = 5
image_size = [128, 128]

model = convAE_test(channels, hiddens, W_shapes, strides, 1, image_size)

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
latest = tf.train.latest_checkpoint('./Parameters/convAE/')
model.saver.restore(sess, latest)
total_embedded = []

for k in image_list:
    images = [(GetInput.getimage(k))]
    recon = sess.run(fetches=[model.recon],
                     feed_dict={model.raw_input: images})

    print(k)
    [embedded] = sess.run(fetches=[model.embedded], feed_dict={model.raw_input: images})
    total_embedded.append(embedded)
    recon = np.squeeze(recon)
    visual.save_image(recon, image_path + '\\AErecon\\', k.split('\\')[-1].split('.')[0])

visual.plot_embedded(total_embedded, image_path + '\\AErecon\\Embedding.png')
