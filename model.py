from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time
from keras.applications import VGG16


def timeit(func):
    def wrapper(*args, **kwargs):
        t = time.clock()
        res = func(*args, **kwargs)
        work_time = time.clock() - t
        return res, work_time
    return wrapper


@timeit
def Conv_model(x_train, y_train, x_val, y_val, x_test, i):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    callbacks = [EarlyStopping(monitor='val_loss', patience=3), ModelCheckpoint(filepath='best_Conv.h5',
                                                                                monitor='val_loss',
                                                                                save_best_only=True)]


    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    model_json = model.to_json()
    with open("Conv.json", 'w') as file:
        file.write(model_json)
    print(model.summary())

    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    print(x_train.shape)
    train_generator = datagen.flow(
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
    pred_generator = datagen.fit(x_val)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=15,
        callbacks=callbacks
    )

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
    plt.close()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(str(i) + "Conv_loss.png")
    plt.close()
    item = x_val[0].reshape(1, x_val[0].shape[0], x_val[0].shape[1], x_val[0].shape[2])
    tmp, time_on_single = make_prediction(model, item)
    return model.predict(x_test), time_on_single


@timeit
def vgg_fine_tune(x_train, y_train, x_val, y_val, x_test, i):
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

    conv_base.summary()

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    callbacks = [EarlyStopping(monitor='val_loss', patience=3), ModelCheckpoint(filepath='best_fine_tuned_vgg.h5',
                                                                                monitor='val_loss',
                                                                                save_best_only=True)]
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )
    print(x_train.shape)
    train_generator = datagen.flow(
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
        steps_per_epoch=100,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=15,
        callbacks=callbacks
    )

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

    return model.predict(x_test), time_on_single


@timeit
def make_prediction(model, item):
    return model.predict(item)
