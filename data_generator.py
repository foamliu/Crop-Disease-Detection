# encoding=utf-8
import json

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical

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
            filename = sample['image_path']
            class_id = sample['class_id']

            image = cv.imread(filename)  # BGR
            image = cv.resize(image, (img_height, img_width), cv.INTER_CUBIC)
            image = image[:, :, ::-1]  # RGB

            if self.usage == 'train':
                image = aug_pipe.augment_image(image)

            batch_inputs[i_batch] = image
            batch_target[i_batch] = to_categorical(class_id, num_classes)

        batch_inputs = preprocess_input(batch_inputs)
        return batch_inputs, batch_target

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)


if __name__ == '__main__':
    data_gen = DataGenSequence('train')
    item = data_gen.__getitem__(0)
    x, y = item

    for i in range(10):
        cv.imwrite('images/sample_{}.jpg'.format(i), x[:, :, ::-1])
