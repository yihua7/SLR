import tensorflow as tf
import numpy as np
import json
import os
import Pre_processing.GetInput as GetInput
from Model.convAE_new import convAE_test
import Model.Grassmann as gs

jsonPath = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\label.json'
imagePath = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\'
classNum = 4

channels = [10, 20, 20, 20, 10]
hiddens = [1024]
W_shapes = [3, 3, 3, 3, 3]
strides = [2, 2, 2, 2, 2]
batch_size = 5
image_size = [128, 128]
k = 16

jsonfile = json.load(open(jsonPath))
sess = tf.Session()
model = convAE_test(channels, hiddens, W_shapes, strides, 1, image_size)
sess.run(tf.global_variables_initializer())
latest = tf.train.latest_checkpoint('./Parameters/convAE/')
model.saver.restore(sess, latest)

label = []
Grassmann = []
for key in jsonfile:
    if os.path.exists(imagePath + jsonfile[key][0]):
        print(key)
        label.append(jsonfile[key][3])
        images = []
        start = int(jsonfile[key][1])
        end = int(jsonfile[key][2])
        for j in range(start, end + 1):
            images.append(GetInput.getimage(imagePath + jsonfile[key][0] + '\\' + str(j) + '.jpeg'))
        count = 0
        length = np.shape(images)[0]
        Grassmann_temp = None
        while count < length:
            next_count = np.min([length, count + batch_size])
            [Grass_sub] = sess.run(fetches=[model.embedded],
                                   feed_dict={model.raw_input: images[count:next_count]})
            if Grassmann_temp is None:
                Grassmann_temp = Grass_sub
            else:
                Grassmann_temp = np.concatenate([Grassmann_temp, Grass_sub], 0)
            count = next_count
        Grassmann_temp = gs.data2grass(Grassmann_temp, k)
        Grassmann.append(Grassmann_temp)

dist = np.zeros([classNum, classNum])
count = np.zeros([classNum, classNum])

for i in range(len(label)):
    for j in range(i+1, len(label)):
        temp = np.linalg.norm(Grassmann[i] - Grassmann[j])
        dist[label[i]][label[j]] += temp
        dist[label[j]][label[i]] += temp
        count[label[i]][label[j]] += 1
        count[label[j]][label[i]] += 1

for i in range(classNum):
    for j in range(classNum):
        if count[i][j] != 0:
            dist[i][j] /= count[i][j]

print(dist)
