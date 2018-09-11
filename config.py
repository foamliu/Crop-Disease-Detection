import os

img_width, img_height = 299, 299
num_channels = 3
train_data = 'data/AgriculturalDisease_trainingset/'
valid_data = 'data/AgriculturalDisease_validationset/'
test_a_data = 'data/AgriculturalDisease_testA/'
train_annot = os.path.join(train_data, 'AgriculturalDisease_train_annotations.json')
valid_annot = os.path.join(valid_data, 'AgriculturalDisease_validation_annotations.json')
train_image_folder = os.path.join(train_data, 'images')
valid_image_folder = os.path.join(valid_data, 'images')
test_a_image_folder = os.path.join(test_a_data, 'images')
num_classes = 61
num_train_samples = 32739
num_valid_samples = 4982
verbose = 1
batch_size = 16
num_epochs = 1000
patience = 50
FREEZE_LAYERS = 2
