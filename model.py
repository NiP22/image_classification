from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time
from keras.applications import VGG16
import numpy as np
from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input




def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    pr = tp / (tp + fp + K.epsilon())
    rec = tp / (tp + fn + K.epsilon())

    f1 = 2 * pr * rec / (pr + rec + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def custom_f1(y_true, y_pred):
    tp = np.sum(y_pred*y_true)
    fp = np.sum((1 - y_true)*y_pred)
    fn = 3*np.sum(y_true*(1 - y_pred))

    pr = tp / (tp + fp + 0.000000001)
    rec = tp / (tp + fn + 0.000000001)

    return 2*pr*rec / (pr + rec + 0.000000001)


def timeit(func):
    def wrapper(*args, **kwargs):
        t = time.clock()
        res = func(*args, **kwargs)
        work_time = time.clock() - t
        return res, work_time
    return wrapper


@timeit
def Conv_model(x_train, y_train, x_val, y_val, i, model_name,
               augmentation=0, vgg_prep=0, batch_norm=0, plot=0):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    callbacks = [EarlyStopping(monitor='val_loss', patience=15), ModelCheckpoint(filepath=(str(i) + model_name + '.h5'),
                                                                                monitor='val_loss',
                                                                                save_best_only=True)]

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])

    if vgg_prep:
        x_val = preprocess_input(x_val)
        x_train = preprocess_input(x_train)
    else:
        x_train /= 255
        x_val /= 255
    if augmentation:
        datagen_augment = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )
    else:
        datagen_augment = ImageDataGenerator()

    datagen = ImageDataGenerator()
    train_generator = datagen_augment.flow(
        x_train,
        y_train,
        shuffle=True,
        batch_size=20
    )
    validation_generator = datagen.flow(
        x_val,
        y_val,
        shuffle=True,
        batch_size=20
    )
    history = model.fit_generator(
        train_generator,
        verbose=1,
        steps_per_epoch=100,#100
        epochs=35,#35
        validation_data=validation_generator,
        validation_steps=15,#15
        callbacks=callbacks
    )
    if plot:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(str(i) + "Conv_acc.png")
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(str(i) + "Conv_loss.png")
        plt.close()
    item = x_val[0].reshape(1, x_val[0].shape[0], x_val[0].shape[1], x_val[0].shape[2])
    tmp, time_on_single = make_prediction(model, item)
    save_model(model, model_name + str(i))
    return model.predict(x_val), time_on_single


@timeit
def vgg_fine_tune(x_train, y_train, x_val, y_val, i, model_name,
                  augmentation=0, vgg_prep=0, batch_norm=0, plot=0):
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(300, 300, 3))
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    callbacks = [EarlyStopping(monitor='val_loss', patience=15), ModelCheckpoint(filepath=(str(i) + model_name + '.h5'),
                                                                                monitor='val_loss',
                                                                                save_best_only=True)]
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    if vgg_prep:
        x_train = preprocess_input(x_train)
        x_val = preprocess_input(x_val)
    else:
        x_train /= 255
        x_val /= 255
    if augmentation:
        datagen_augment = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )
    else:
        datagen_augment = ImageDataGenerator()
    train_generator = datagen_augment.flow(
        x_train,
        y_train,
        shuffle=True,
        batch_size=20
    )

    datagen = ImageDataGenerator()
    validation_generator = datagen.flow(
        x_val,
        y_val,
        shuffle=True,
        batch_size=20
    )

    history = model.fit_generator(
        train_generator,
        verbose=2,
        steps_per_epoch=100,#100
        epochs=30,#30
        validation_data=validation_generator,
        validation_steps=15,#15
        callbacks=callbacks
    )
    if plot:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(str(i) + "VGG_fine_tune_acc.png")
        plt.close()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(str(i) + "VGG_fine_tune_loss.png")
        plt.close()

    item = x_val[0].reshape(1, x_val[0].shape[0], x_val[0].shape[1], x_val[0].shape[2])
    tmp, time_on_single = make_prediction(model, item)
    save_model(model, model_name + str(i))
    return model.predict(x_val), time_on_single


@timeit
def make_prediction(model, item):
    return model.predict(item)


def save_model(model, name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    model.save_weights(name + ".h5")