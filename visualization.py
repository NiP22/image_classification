from keras.models import load_model, model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from keras import models
from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras.backend as K
from keras.applications.vgg16 import preprocess_input
from image_classification.directory_work import get_train_data
from sklearn.metrics import roc_curve, roc_auc_score
from keras import models
from keras import layers
from keras import optimizers




img_path = r'C:\Users\Pavel.Nistsiuk\PycharmProjects\people_class\try.png'
img = image.load_img(img_path, target_size=(300, 300))
img_tensor = image.img_to_array(img)
img_tensor /= 255


json_file = open('conv_noAugment_simplePrep_noBN.json')
load_json = json_file.read()
model = model_from_json(load_json)
model.load_weights('conv_noAugment_simplePrep_noBN.h5')
img_tensor = img_tensor.reshape((1, 300, 300, 3))

layer_outputs = [layer.output for layer in model.layers[:8]]

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
for i in range(32):
    plt.matshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.savefig(r'C:\Users\Pavel.Nistsiuk\PycharmProjects\people_class\\vis_plots\\' + str(i) + '.jpg')
    plt.close()

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig(r'C:\Users\Pavel.Nistsiuk\PycharmProjects\people_class\\vis_plots\\' + layer_name + '.jpg')
    plt.close()

plt.show()