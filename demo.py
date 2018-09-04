# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input

from config import train_data, test_a_data
from model import build_model
from utils import get_best_model

if __name__ == '__main__':
    best_model, epoch = get_best_model()
    model = build_model()
    model.load_weights(best_model)

    labels = [folder for folder in os.listdir(train_data) if os.path.isdir(os.path.join(train_data, folder))]

    test_images = [f for f in os.listdir(test_a_data) if
                   os.path.isfile(os.path.join(test_a_data, f)) and f.endswith('.jpg')]
    num_samples = 20
    samples = random.sample(test_images, num_samples)

    if not os.path.exists('images'):
        os.makedirs('images')

    results = []
    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_a_data, image_name)
        print('Start processing image: {}'.format(filename))
        image = cv.imread(filename)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
        rgb_img = preprocess_input(rgb_img)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = int(np.argmax(preds))
        print(labels[class_id])
        results.append({'label': labels[class_id], 'prob': '{:.4}'.format(prob)})
        cv.imwrite('images/{}_out.png'.format(i), image)

    print(results)
    with open('results.json', 'w') as file:
        json.dump(results, file)

    K.clear_session()
