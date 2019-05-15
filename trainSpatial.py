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
model = Spatial_hourglass(block_number=block_number, layers=layers, out_dim=out_dim, point_num=point_num, lr=2.5e-4,
                           training=True, dropout_rate=dropout_rate)

cycle = 10000

image_path = data_set_path + 'JPEG\\'
label_path = data_set_path + 'hands_annotation\\'
model.train(data_path=image_path,
            label_path=label_path,
            batch_size=batch_size,
            maxepoch=cycle+1,
            continue_train=False,
            base=0,
            step='all')
