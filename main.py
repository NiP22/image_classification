import matplotlib.pyplot as plt
from image_classification.model import Conv_model, vgg_fine_tune
from sklearn.metrics import roc_curve, roc_auc_score
from image_classification.create_table import add_stats_to_csv
import numpy as np
from image_classification.directory_work import get_train_data
from statsmodels.stats.weightstats import _tconfint_generic
from math import sqrt
import pandas as pd


def kfold_on_model(model, model_name, x_train, y_train, augmentation=0, vgg_prep=0, batch_norm=0):
    k = 4
    fold_size = y_train.shape[0] // (k + 1)
    X_test = x_train[:fold_size]
    Y_test = y_train[:fold_size]
    auc_scores = list()
    for i in range(1, k):#!!!!
        X_t = np.concatenate((x_train[fold_size:fold_size * i], x_train[fold_size * (i + 1):]))
        Y_t = np.concatenate((y_train[fold_size:fold_size * i], y_train[fold_size * (i + 1):]))
        X_v = x_train[fold_size * i:fold_size * (i + 1)]
        Y_v = y_train[fold_size * i:fold_size * (i + 1)]
        count = 0
        for each in Y_t:
            if each == 1:
                count += 1
        count = 0
        for each in Y_v:
            if each == 1:
                count += 1
        (pred, learning_time), time_on_single = model(X_t, Y_t, X_v, Y_v, X_test, i, model_name, augmentation,
                                                      vgg_prep, batch_norm)
        fpr, tpr, thresholds = roc_curve(Y_test, pred)
        auc_on_model = roc_auc_score(Y_test, pred)
        auc_scores.append(auc_on_model)
        add_stats_to_csv(model_name, auc_on_model, time_on_single, learning_time, i)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.savefig(str(i) + model_name + "_roc_curve.png")
        plt.close()
    auc_scores = np.array(auc_scores)
    mean_std = auc_scores.std(ddof=1) / sqrt(len(auc_scores))
    beg, end = _tconfint_generic(auc_scores.mean(), mean_std, len(auc_scores) - 1, 0.05, 'two-sided')
    df = pd.read_csv('Confidence intervals.csv')
    new_item = pd.DataFrame([[model_name, beg, end]],
                            columns=['Name', 'Start_of_interval', 'End_of_interval'])
    df = pd.concat([df, new_item])
    df.to_csv("Confidence intervals.csv", index=False)


base_dir = r'D:\python_projects\workStuff'

x_train, y_train = get_train_data(base_dir)
indexes = np.arange(y_train.shape[0])
np.random.shuffle(indexes)
for i_old, i_new in enumerate(indexes):
    x_train[i_old], y_train[i_old] = x_train[i_new], y_train[i_new]


kfold_on_model(vgg_fine_tune, "vgg_noAugment_simplePrep_noBN", x_train, y_train,
               augmentation=0, vgg_prep=0, batch_norm=0)
kfold_on_model(Conv_model, "conv_noAugment_simplePrep_noBN", x_train, y_train,
               augmentation=0, vgg_prep=0, batch_norm=0)

kfold_on_model(vgg_fine_tune, "vgg_Augment_simplePrep_noBN", x_train, y_train,
               augmentation=1, vgg_prep=0, batch_norm=0)
kfold_on_model(Conv_model, "conv_Augment_simplePrep_noBN", x_train, y_train,
               augmentation=1, vgg_prep=0, batch_norm=0)

kfold_on_model(vgg_fine_tune, "vgg_noAugment_vggPrep_noBN", x_train, y_train,
               augmentation=0, vgg_prep=1, batch_norm=0)
kfold_on_model(Conv_model, "conv_noAugment_vggPrep_noBN", x_train, y_train,
               augmentation=0, vgg_prep=1, batch_norm=0)

kfold_on_model(vgg_fine_tune, "vgg_noAugment_simplePrep_BN", x_train, y_train,
               augmentation=0, vgg_prep=0, batch_norm=1)
kfold_on_model(Conv_model, "conv_noAugment_simplePrep_BN", x_train, y_train,
               augmentation=0, vgg_prep=0, batch_norm=1)

kfold_on_model(vgg_fine_tune, "vgg_Augment_vggPrep_noBN", x_train, y_train,
               augmentation=1, vgg_prep=1, batch_norm=0)
kfold_on_model(Conv_model, "conv_Augment_vggPrep_noBN", x_train, y_train,
               augmentation=1, vgg_prep=1, batch_norm=0)

kfold_on_model(vgg_fine_tune, "vgg_noAugment_vggPrep_BN", x_train, y_train,
               augmentation=0, vgg_prep=1, batch_norm=1)
kfold_on_model(Conv_model, "conv_noAugment_vggPrep_BN", x_train, y_train,
               augmentation=0, vgg_prep=1, batch_norm=1)

kfold_on_model(vgg_fine_tune, "vgg_Augment_simplePrep_BN", x_train, y_train,
               augmentation=1, vgg_prep=0, batch_norm=1)
kfold_on_model(Conv_model, "conv_Augment_simplePrep_BN", x_train, y_train,
               augmentation=1, vgg_prep=0, batch_norm=1)

kfold_on_model(vgg_fine_tune, "vgg_Augment_vggPrep_BN", x_train, y_train,
               augmentation=1, vgg_prep=1, batch_norm=1)
kfold_on_model(Conv_model, "conv_Augment_vggPrep_BN", x_train, y_train,
               augmentation=1, vgg_prep=1, batch_norm=1)
