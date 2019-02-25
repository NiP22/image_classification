import matplotlib.pyplot as plt
from image_classification.model import model
from sklearn.metrics import roc_curve, auc
from image_classification.create_table import add_stats_to_csv
import numpy as np
from image_classification.directory_work import get_train_data





base_dir = r'C:\Users\Pavel.Nistsiuk\PycharmProjects\people_class'

x_train, y_train = get_train_data(base_dir)

indexes = np.arange(y_train.shape[0])
np.random.shuffle(indexes)
for i_old, i_new in enumerate(indexes):
    x_train[i_old], y_train[i_old] = x_train[i_new], y_train[i_new]

fold_size = y_train.shape[0] // 5

file = open("metrics.txt", 'w')


for i in range(5):
    X_t = np.concatenate((x_train[:fold_size * i], x_train[fold_size * (i + 1):]))
    Y_t = np.concatenate((y_train[:fold_size * i], y_train[fold_size * (i + 1):]))
    X_v = x_train[fold_size * i:fold_size * (i + 1)]
    Y_v = y_train[fold_size * i:fold_size * (i + 1)]

    pred, time = model(X_t, Y_t, X_v, Y_v, i)
    fpr, tpr, treshholds = roc_curve(Y_v, pred)
    auc_model = auc(fpr, tpr)
    print(str(auc_model))
    file.write(str(auc_model) + '\n')
    add_stats_to_csv("Conv32 64 128", auc_model, time, i)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.savefig(str(i) + "_roc_curve.png")
    plt.close()
file.close()

