import matplotlib.pyplot as plt
from image_classification.model import Conv_model, vgg_fine_tune
from sklearn.metrics import roc_curve, roc_auc_score
from image_classification.create_table import add_stats_to_csv
import numpy as np
from image_classification.directory_work import get_train_data
from statsmodels.stats.weightstats import _tconfint_generic
from math import sqrt
import pandas as pd
from sklearn.model_selection import StratifiedKFold



def kfold_on_model(model, model_name, x_train, y_train, augmentation=0, vgg_prep=0, batch_norm=0):
    auc_scores = list()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    for i, (train, test) in enumerate(kfold.split(x_train, y_train)):
        X_t = x_train[train]
        Y_t = y_train[train]
        X_v = x_train[test]
        Y_v = y_train[test]
        count = 0
        for each in Y_t:
            if each == 1:
                count += 1
        count = 0
        for each in Y_v:
            if each == 1:
                count += 1
        (pred, learning_time), time_on_single = model(X_t, Y_t, X_v, Y_v, i, model_name, augmentation,
                                                      vgg_prep, batch_norm)
        fpr, tpr, thresholds = roc_curve(Y_v, pred)
        auc_on_model = roc_auc_score(Y_v, pred)
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


base_dir = r'C:\Users\Pavel.Nistsiuk\PycharmProjects\people_class'

x_train, y_train = get_train_data(base_dir)
indexes = np.arange(y_train.shape[0])
np.random.shuffle(indexes)
for i_old, i_new in enumerate(indexes):
    x_train[i_old], y_train[i_old] = x_train[i_new], y_train[i_new]



kfold_on_model(Conv_model, "conv_noAugment_simplePrep_noBN", x_train, y_train,
               augmentation=1, vgg_prep=1, batch_norm=1)

