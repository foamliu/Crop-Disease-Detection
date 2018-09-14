# import the necessary packages
# coding:utf-8
import json
import os

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm

from config import train_data, test_a_image_folder, img_height, img_width
from model import build_model
from utils import get_best_model

if __name__ == '__main__':
    best_model, epoch = get_best_model()
    model = build_model()
    model.load_weights(best_model)

    labels = [folder for folder in os.listdir(train_data) if os.path.isdir(os.path.join(train_data, folder))]

    test_images = [f for f in os.listdir(test_a_image_folder) if
                   os.path.isfile(os.path.join(test_a_image_folder, f)) and f.lower().endswith('.jpg')]

    results = []
    for image_id in tqdm(test_images):
        filename = os.path.join(test_a_image_folder, image_id)
        # print('Start processing image: {}'.format(filename))
        image = cv.imread(filename)
        image = cv.resize(image, (img_height, img_width), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
        rgb_img = preprocess_input(rgb_img)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = int(np.argmax(preds))
        # print(labels[class_id])
        results.append({'image_id': image_id, 'disease_class': class_id})

    with open('eval.json', 'w') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    K.clear_session()
