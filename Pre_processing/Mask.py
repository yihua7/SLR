import cv2
import os
import matplotlib.pyplot as plt


def detect_and_draw(filename, image_detct, image_draw, color):
    classifier_face = cv2.CascadeClassifier(filename)
    faces = classifier_face.detectMultiScale(image_detct)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_draw, (x, y), (x + w, y + h), color, 2)


def Mask(target, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    name_array = target.split('/')
    image_name = name_array[-1]
    print("Processing" + image_name + "\n")
    saver = save_path + image_name
    image = cv2.imread(target)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    detect_and_draw('haarcascade_frontalface_default.xml', image_gray, image, (255, 255, 255))
    detect_and_draw('hand.xml', image_gray, image, (255, 255, 255))
    detect_and_draw('fist.xml', image_gray, image, (255, 255, 255))
    detect_and_draw('haarcascade_frontalface_default.xml', image_gray, image, (255, 255, 255))
    detect_and_draw('fist_v3.xml', image_gray, image, (255, 255, 255))
    detect_and_draw('aGest.xml', image_gray, image, (255, 255, 255))

    # plt.imshow(image)
    # plt.show()
    cv2.imwrite(saver, image)


def Batch_Mask(target_path, save_path):
    images = sorted(os.listdir(target_path))
    for image in images:
        if image.endswith('.jpg') or image.endswith('.jpeg'):
            Mask(target_path+'/'+image, save_path)


# Batch_Mask("../Data/raw/scene2-camera1/", "../Data/processed/scene2-cameral1/")

Batch_Mask("../Data/ASL/JPEG/scene2-camera1.vid/", "../Data/processed/scene2-camera1.vid/")