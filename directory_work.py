import os
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt


def to_grayscale(img_arr):
    return (0.21 * img_arr[:, :, :1]) + (0.72 * img_arr[:, :, 1:2]) + (0.07 * img_arr[:, :, -1:])


def get_train_data(base_dir):
    train_dir = os.path.join(base_dir, 'Train')
    train_pos_dir = os.path.join(train_dir, 'pos')
    train_neg_dir = os.path.join(train_dir, 'neg')
    fnames = [os.path.join(train_neg_dir, fname) for fname in os.listdir(train_neg_dir)]
    x = list()
    y = list()
    for img_path in fnames:
        tmp = image.img_to_array(image.load_img(img_path, target_size=(300, 300)))
        x.append(tmp)
        y.append(0)
    fnames = [os.path.join(train_pos_dir, fname) for fname in os.listdir(train_pos_dir)]
    for img_path in fnames:
        tmp = image.img_to_array(image.load_img(img_path, target_size=(300, 300)))
        x.append(tmp)
        y.append(1)
    return np.array(x), np.array(y)


def get_train_data_gray(base_dir):
    train_dir = os.path.join(base_dir, 'Train')
    train_pos_dir = os.path.join(train_dir, 'pos')
    train_neg_dir = os.path.join(train_dir, 'neg')
    fnames = [os.path.join(train_neg_dir, fname) for fname in os.listdir(train_neg_dir)]
    x = list()
    y = list()
    for img_path in fnames:
        tmp = image.img_to_array(image.load_img(img_path, target_size=(300, 300)))
        tmp = to_grayscale(tmp)
        x.append(tmp)
        y.append(0)
    fnames = [os.path.join(train_pos_dir, fname) for fname in os.listdir(train_pos_dir)]
    for img_path in fnames:
        tmp = image.img_to_array(image.load_img(img_path, target_size=(300, 300)))
        tmp = to_grayscale(tmp)
        x.append(tmp)
        y.append(1)
    return np.array(x), np.array(y)

