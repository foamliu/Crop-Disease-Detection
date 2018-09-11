from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

from config import img_height, img_width, num_classes, FREEZE_LAYERS


def build_model():
    base_model = InceptionResNetV2(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    return model
