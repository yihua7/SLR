import numpy as np
import tensorflow as tf
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt
import json


def down_sample(image, size):
    image = tf.image.resize_images(image, size, method=2)
    image = tf.cast(image, tf.int32)
    return image


def getimage(path):
    return np.array(Image.open(path), dtype=float)


def getlabel(path):
    reader = csv.reader(open(path, 'r'))
    anno = []
    for row in reader:
        anno.append(row)
    return np.array(anno)


def getlabel_batch(labeldata, batch_size, start_data):
    label = []
    batch = []
    index = start_data
    upper = np.shape(labeldata)[0]
    for i in range(batch_size):
        if index >= upper:
            break
        number = int(labeldata[index][0])
        batch.append(number)
        while index < upper and number == int(labeldata[index][0]):
            label.append(labeldata[index])
            index += 1
    return label, index, batch


def getheatmap(label, size):
    heatmap = []
    number = int(label[0][0])
    temp = np.zeros(size, dtype=float)
    for line in label:
        newnumber = int(line[0])
        if newnumber != number:
            heatmap.append(temp)
            temp = np.zeros(size, dtype=float)
            number = newnumber
        y = int(line[2])
        x = int(line[3])
        h = int(line[4])
        w = int(line[5])
        for i in range(x, x+w):
            for j in range(y, y+h):
                temp[i][j][int(line[1])-1] = 1.
    heatmap.append(temp)
    return heatmap


def csv2json(csv_path, json_path):
    dict = {"TWENTY": 0, "ALONE": 1, "ONE": 2, "ALWAYS": 3}
    # Input csv file path
    store ={}
    category = set()
    count = {}
    i = 0
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=",")
        words = ""
        for line in reader:
            if len(line[0]) > 0:
                words = line[0]
            elif words == 'TWENTY' or words == 'ALONE' or words == 'ONE' or words == 'ALWAYS':
                if line[1].startswith('-'):
                    continue
                i += 1

                label = dict[words]
                store[words+str(i)] = [line[11]+'\\scene'+line[12]+'-camera1.vid', line[13], line[14], label]

    for r in count:
        if count[r] > 1:
            print(r, count[r])
    with open(json_path, 'w') as outfile:
        json.dump(store, outfile)


if __name__ == '__main__':
    csv2json('../Data/ASL/label.csv', '../Data/ASL/label.json')
