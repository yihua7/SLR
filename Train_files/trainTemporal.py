from Model.TemporalAttention import SelfAttention

channel = [16, 128, 512]
convd = 5
W_shape = 5
fdim = 64
hdim = 32
classNumber = 4
model = SelfAttention(channel, convd, W_shape, fdim, hdim, classNumber, 1e-4)

jsonPath = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\label.json'
imagePath = 'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\'

model.train(jsonPath, imagePath, 50000, False)
