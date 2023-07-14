import cv2
import numpy as np
import os
from PIL import Image

# PATH TO DIR
path = os.path.dirname(os.path.abspath(__file__))


recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

data_path = path+r'/persons'

def get_img_labels(data_path):
    # PATH TO FACES
    img_paths = []
    for folder in os.listdir(data_path):
        for pic in os.listdir(data_path + '/' + folder):
            img_paths.append(os.path.join(data_path + '/' + folder + '/' + pic))
    imgs = []
    labels = []
    # GO THROUGHT IMGS IN DIR
    for img_path in img_paths:
        print(img_path)
        img_pil = Image.open(img_path).convert('L')
        # IMG -> NUMPY
        img = np.array(img_pil, 'uint8')
        # GETING USER FROM path
        path_list = img_path.split('/')
        user_id = int(path_list[-2].split(':')[0])
        # FACE DETECTION
        faces = face_cascade.detectMultiScale(img)
        # if face
        for x, y, w, h in faces:
            # ADDING FACE TO FACE LIST
            imgs.append(img[y: y + h, x: x + w])
            # ADDING USER_ID TO LABEL LIST
            labels.append(user_id)
            # SHOW CURRENT PIC
            cv2.imshow('Adding faces training...', img[y: y + h, x: x + w])
            cv2.waitKey(100)
    return imgs, labels

images, labels = get_img_labels(data_path)

#TRAINING
recognizer.train(images, np.array(labels))

# SAVING RESULT
trainer_path = os.path.join(path, 'model')
if not os.path.isdir(trainer_path):
    os.mkdir(trainer_path)
recognizer.save(os.path.join(trainer_path, 'trainer.yml'))
# CLOSE ALL WINDOWS
cv2.destroyAllWindows()