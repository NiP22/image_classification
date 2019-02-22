import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image


def model(x_train, y_train, x_val, y_val, i):
    '''
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    '''
    model = models.Sequential()

    model.add(layers.Dense(32, input_shape=(300, 300, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

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
    print(train_generator.next())
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
        epochs=1,
        validation_data=validation_generator,
        validation_steps=15
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
    plt.savefig(str(i) + "model_acc.png")
    plt.close()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(str(i) + "model_loss.png")
    plt.close()
    #return model.predict_generator(pred_generator)
    return model.predict(x_val)
