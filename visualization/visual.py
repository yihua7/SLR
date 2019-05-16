from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import csv
import cv2


def add_color(image, x, y):
    image[x, y, 0] = 255
    image[x, y, 1] = 255
    image[x, y, 2] = 255
    return image


def draw_line(image, cdi_one, cdi_two, line_width):
    image = np.array(image)
    w, h, _ = list(np.shape(image))
    if cdi_one[0] < 0 or cdi_one[0] >= w or\
        cdi_one[1] < 0 or cdi_two[1] >= h or\
        cdi_two[0] < 0 or cdi_two[0] >= w or\
        cdi_two[1] < 0 or cdi_two[1] >= h:
        return image
    gap_x = int(cdi_one[0]-cdi_two[0])
    gap_y = int(cdi_one[1]-cdi_two[1])
    signx = int(np.sign(gap_x))
    signy = int(np.sign(gap_y))
    if gap_x != 0:
        for i in range(0, abs(gap_x)):
            x = int(cdi_two[0] + i * signx)
            y = int(cdi_two[1] + abs(i*gap_y/gap_x)*signy)
            for j in range(0, line_width):
                yy = max(min(y + j - int(line_width/2), h-1), 0)
                image = add_color(image, int(x), int(yy))
    else:
        for i in range(0, abs(gap_y)):
            x = int(cdi_one[0])
            y = int(cdi_two[1] + i*np.sign(gap_y))
            for j in range(0, line_width):
                yy = max(min(y + j - int(line_width/2), h-1), 0)
                image = add_color(image, int(x), int(yy))
    return np.array(image)


def draw_annotation(annotation, video_file, save_path):
    # Get Annotation
    print("Loading Annotation")
    reader = csv.reader(open(annotation, 'r'))
    anno = []
    for row in reader:
        anno.append(row)

    # Draw Annotation
    print("Drawing")
    oldindex = -1
    for i in range(np.shape(anno)[0]):
        for j in range(np.shape(anno)[1]):
            anno[i][j] = int(anno[i][j])
        print("%dth image, %dth frame" % (i, anno[i][0]))
        p = np.zeros([2])
        p[1] = anno[i][2]
        p[0] = anno[i][3]
        h = anno[i][4]
        w = anno[i][5]
        if oldindex != anno[i][0]:
            img = Image.open(video_file + '\\' + str(anno[i][0]) + '.jpeg')
            image = np.array(img.getdata())
            image = image.reshape([img.height, img.width, 3])
            oldindex = anno[i][0]
        image = draw_line(image, p, p+[0, h], 10)
        image = draw_line(image, p, p + [w, 0], 10)
        image = draw_line(image, p + [w, h], p + [0, h], 10)
        image = draw_line(image, p + [w, h], p + [w, 0], 10)
        img = Image.fromarray(image.astype('uint8')).convert('RGB')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img.save(save_path + '\\' + str(anno[i][0]) + ".jpg")


def save_image(image, path, name):
    image = np.array(image)
    img = Image.fromarray(image.astype('uint8')).convert('RGB')
    if not os.path.exists(path):
        os.makedirs(path)
    img.save(path + '\\' + name + ".jpg")


def plot_info(loss, step, name=''):
    plt.close('all')
    plt.plot(step, loss, "b.-")
    plt.title("Spatial Attention Info")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.savefig("train_info"+name+".png")
    plt.show()


def plot_loss(loss, KL, recon, step):
    plt.close('all')
    plt.plot(step, loss, "b.-")
    plt.plot(step, KL, "r.-")
    plt.plot(step, recon, "g.-")
    plt.title("Loss Info of convVAE")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("./visualization/VAE_train_info.png")
    plt.show()


def plot_AE_loss(loss, step):
    plt.close('all')
    plt.plot(step, loss, "b.-")
    plt.title("Loss Info of convAE")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("./visualization/AE_train_info.png")
    plt.show()


def hotmap_visualization(heatmap, image, labelimage, path, name):
    heatmap = np.array(heatmap)
    image = np.squeeze(np.array(image, int))
    labelimage = np.squeeze(np.array(labelimage))
    attention = np.max(heatmap, 2)
    image_attention = copy.copy(image)
    for i in range(np.shape(image)[-1]):
        image_attention[:, :, i] = np.multiply(image_attention[:, :, i], attention)
    plt.close('all')
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.subplot(2, 2, 2)
    plt.imshow(heatmap)
    plt.subplot(2, 2, 3)
    plt.imshow(np.array(image_attention, int))
    plt.subplot(2, 2, 4)
    plt.imshow(labelimage)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+name)
    plt.show()


def spatial_output(heatmap, image, path, name):
    heatmap = np.array(heatmap)
    image = np.squeeze(np.array(image, int))
    attention = np.max(heatmap, 2)
    for i in range(np.shape(image)[-1]):
        image[:, :, i] = np.multiply(image[:, :, i], attention)
#    plt.close('all')
#    plt.axis('off')
#    plt.imshow(image)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = name.split('\\')[-1]
    cv2.imwrite(path+filename, image)
#    plt.savefig(path+filename, bbox_inches='tight')

'''
draw_annotation('D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\hands_annotation\\scene2-camera1.csv',
                'D:\\UserData\\DeepLearning\\Sign-Language-Recognition\\Data\\ASL\\JPEG\\scene2-camera1.vid',
                '..\\Data\\ASL\\Test')
'''
