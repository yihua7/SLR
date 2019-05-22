import tensorflow as tf
import glob
from Pre_processing import GetInput
from visualization import visual
import numpy as np

image_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\ASL_2007_05_24\\scene10-camera1.vid'
num = 128

raw = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='raw')
image = tf.image.resize_images(raw, [num, num], method=2)

image_list = glob.glob(image_path + '\\*.jpeg')
sess = tf.Session()

for k in image_list:
    images = [(GetInput.getimage(k))]
    recon = sess.run(fetches=[image],
                     feed_dict={raw: images})

    print(k)
    recon = np.squeeze(recon)
    visual.save_image(recon, image_path + '\\' + str(num) + '\\', k.split('\\')[-1].split('.')[0])