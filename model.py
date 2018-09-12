from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.models import Model

from config import img_height, img_width, num_channels, num_classes, FREEZE_LAYERS, dropout_rate


def build_model():
    base_model = InceptionResNetV2(input_shape=(img_height, img_width, num_channels), weights='imagenet',
                                   include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate, name='Dropout')(x)
    x = Dense(num_classes, name='Logits')(x)
    x = Activation('softmax', name='Predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    return model
