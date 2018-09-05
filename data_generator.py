# encoding=utf-8
import json
import os

import cv2 as cv
import dlib
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence

from augmentor import aug_pipe
from config import img_height, img_width, batch_size, num_classes



class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        if self.usage == 'train':
            gt_file = 'train_gt_file.json'
        else:
            gt_file = 'valid_gt_file.json'

        with open(gt_file, 'r') as file:
            self.samples = json.load(file)


    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        batch_inputs = np.empty((3, length, img_height, img_width, 3), dtype=np.float32)
        batch_target = np.zeros((length, num_classes), dtype=np.float32)

        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            for j, role in enumerate(['a', 'p', 'n']):
                image_name = sample[role]
                filename = os.path.join(self.image_folder, image_name)
                image = cv.imread(filename)  # BGR
                image = image[:, :, ::-1]  # RGB
                dets = self.detector(image, 1)

                num_faces = len(dets)
                if num_faces > 0:
                    # Find the 5 face landmarks we need to do the alignment.
                    faces = dlib.full_object_detections()
                    for detection in dets:
                        faces.append(self.sp(image, detection))
                    image = dlib.get_face_chip(image, faces[0], size=img_size)
                else:
                    image = cv.resize(image, (img_size, img_size), cv.INTER_CUBIC)

                if self.usage == 'train':
                    image = aug_pipe.augment_image(image)

                batch_inputs[j, i_batch] = preprocess_input(image)

        return batch_inputs, batch_target

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)


if __name__ == '__main__':
    data_gen = DataGenSequence('train')
    item = data_gen.__getitem__(0)
    x, y = item
    a = revert_pre_process(x[0])
    p = revert_pre_process(x[1])
    n = revert_pre_process(x[2])
    for i in range(10):
        cv.imwrite('images/sample_a_{}.jpg'.format(i), a[i][:, :, ::-1])
        cv.imwrite('images/sample_p_{}.jpg'.format(i), p[i][:, :, ::-1])
        cv.imwrite('images/sample_n_{}.jpg'.format(i), n[i][:, :, ::-1])
