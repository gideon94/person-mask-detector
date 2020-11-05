# -*- coding: utf-8 -*-


import os

import numpy as np
import tensorflow as tf
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def get_model():
    baseModel = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                                                  input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))
    base_model = baseModel.output
    base_model = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(base_model)
    base_model = tf.keras.layers.Flatten(name="flatten")(base_model)
    base_model = tf.keras.layers.Dense(128, activation="relu")(base_model)
    base_model = tf.keras.layers.Dropout(0.5)(base_model)
    base_model = tf.keras.layers.Dense(2, activation="softmax")(base_model)
    model = tf.keras.models.Model(inputs=baseModel.input, outputs=base_model)
    for layer in baseModel.layers:
        layer.trainable = False
    es = tf.keras.callbacks.EarlyStopping(patience=2)

    opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4 / 20)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    return model


def train():
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images("train"))
    if len(imagePaths) == 0:
        raise Exception(
            "Train images not found!. Please verify the download path")
    data = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        # load the input image (224x224) and preprocess it
        image = tf.keras.preprocessing.image.load_img(
            imagePath, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        # update the data and labels lists, respectively
        data.append(image)

        labelC = 0 if label == "without_mask" else 1
        labels.append(labelC)
    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    np.unique(labels)

    labels = tf.keras.utils.to_categorical(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)
    es = tf.keras.callbacks.EarlyStopping(patience=2)

    model.fit(trainX, trainY, steps_per_epoch=32, validation_data=(
        testX, testY), validation_steps=len(testX)/32, epochs=20, callbacks=[es])

    model.save("masknet/mask_weights/mask_model.h5")

    model.save_weights("masknet/mask_weights/mask_weights.h5")


def evaluate():
    imagePaths = list(paths.list_images("test"))
    dataTest = []
    labelsTest = []

    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        image = tf.keras.preprocessing.image.load_img(
            imagePath, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        # update the data and labels lists, respectively
        dataTest.append(image)
        labelC = 0 if label == "without_mask" else 1
        labelsTest.append(labelC)

    dataTest = np.array(dataTest, dtype="float32")
    labelsTest = np.array(labelsTest)

    labelsTest_B = tf.keras.utils.to_categorical(labelsTest)

    predIdxs = model.predict(dataTest, batch_size=32)

    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(labelsTest_B.argmax(axis=1), predIdxs,
                                target_names=['With', 'Without']))


model = get_model()

train()

evaluate()
