import pandas as pd
import time


def add_stats_to_csv(model_name, model_acc, time_on_single, learning_time, iter):
    table = pd.read_csv("Stats.csv")
    new_item = pd.DataFrame([[model_name, model_acc, time_on_single, learning_time, iter]],
                            columns=['Name', 'Accuracy', 'Learning_time', 'Predict_time', 'Iteration'])
    table = pd.concat([table, new_item])
    table.to_csv("Stats.csv", index=False)
