import json
import os
from Model.SpatialAttention import Spatial_hourglass
data_set_path = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\'
block_number = 8
layers = 3
lr = 2e-4
out_dim = 256
point_num = 3
maxepoch = 300001
dropout_rate = 0.2

batch_size = 5

# Create Model
model = Spatial_hourglass(block_number=block_number, layers=layers, out_dim=out_dim, point_num=point_num, lr=2.5e-4)
jsonpath = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\label.json'
image_path = data_set_path + 'JPEG\\'
jsonfile = json.load(open(jsonpath))

# model.test(image_path + '\\ASL_2008_05_12a\\scene3-camera1.vid\\', mode='origin')

for key in jsonfile:
    print(image_path + jsonfile[key][0] + '\\spatial\\downSample')
    if os.path.exists(image_path + jsonfile[key][0]) and not os.path.exists(image_path + jsonfile[key][0] + '\\spatial\\downSample'):
        print(jsonfile[key][0])
        model.test(image_path + jsonfile[key][0], mode='downSample')
        model.test(image_path + jsonfile[key][0], mode='origin')

