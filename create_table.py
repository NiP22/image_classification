import pandas as pd
import time

def add_stats_to_csv(model_name, model_acc, model_time, iter):
    table = pd.read_csv("Stats.csv")
    new_item = pd.DataFrame([[model_name, model_acc, model_time, iter]],
                            columns=['Name', 'Accuracy', 'Time', 'Iteration'])
    table = pd.concat([table, new_item])
    table.to_csv("Stats.csv", index=False)
