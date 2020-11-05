import os

import numpy as np
import tensorflow as tf
from django.conf import settings

model_dir = getattr(settings, "MODEL_ROOT", None)

if not model_dir:
    raise "Model Not found. Please verify MODEL_ROOT settings"


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def get_mask_model():
    base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                                                   input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))
    mask_model = base_model.output
    mask_model = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(mask_model)
    mask_model = tf.keras.layers.Flatten(name="flatten")(mask_model)
    mask_model = tf.keras.layers.Dense(128, activation="relu")(mask_model)
    mask_model = tf.keras.layers.Dropout(0.5)(mask_model)
    mask_model = tf.keras.layers.Dense(2, activation="softmax")(mask_model)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=mask_model)

    for layer in base_model.layers:
        layer.trainable = False
    opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4 / 20)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    model.load_weights(os.path.join(model_dir, "mask_weights.h5"))
    return model


mask_net = get_mask_model()


def get_pred_mask(image):
    # t = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    # t = tf.keras.preprocessing.image.img_to_array(t)
    t = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    y = mask_net.predict(np.array([t]))
    label = ["No Mask", "Has Mask"]
    resp = list()
    for _y in y:
        resp.append(
            {'label': label[_y.argmax(axis=-1)], 'confidence': max(_y)})
    return resp


if __name__ == "__main__":
    print(get_pred_mask("/home/madhan/person-mask-tf/masknet/media/n_1.jpeg"))
