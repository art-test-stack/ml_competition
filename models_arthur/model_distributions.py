import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

import sys
sys.path.append('../')
import utils


def normalize_df(df, keys, time_column):
    df_normalized = df.copy()
    columns_to_drop = [ c for c in df_normalized.columns if (c not in keys) and (c != time_column)]
    df_normalized = df_normalized.drop(columns=columns_to_drop)
    for key in keys:
        df_normalized[key] = (df[key] - df[key].mean()) / df[key].std()
    return df_normalized

# USEFULL DATA

keys = utils.get_most_important_keys()
keys.remove('date_forecast') if 'date_forecast' in keys else None

days = utils.get_days_to_predict(diff_path='../')
train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c = utils.read_files(diff_path='../')

X_tr_est_a = normalize_df(X_train_estimated_a, keys, 'date_forecast')
X_tr_obs_a = normalize_df(X_train_observed_a, keys, 'date_forecast')
X_te_est_a = normalize_df(X_test_estimated_a, keys, 'date_forecast')

list_to_plot = [X_train_estimated_a, X_train_observed_a, X_test_estimated_a, X_tr_est_a, X_tr_obs_a, X_te_est_a]
plot_titles = ['X_train_estimated_a', 'X_train_observed_a', 'X_test_estimated_a', 'X_tr_est_a', 'X_tr_obs_a', 'X_te_est_a']

def plot_normal_distribution(df, mean, std):
    plt.hist(df, bins=30, density=True, alpha=0.6, color='b')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    plt.plot(x, pdf, 'k')

def subplot_normal_distrib(list_to_plot = list_to_plot, plot_titles = plot_titles, keys=keys, dims = (2, 3)):
    assert len(list_to_plot) == dims[0] * dims[1]
    assert len(plot_titles) == len(list_to_plot)
    for key in keys:
        plt.figure(figsize=(20, 8))
        for k in range(6):
            plt.subplot(dims[0], dims[1], k + 1)
            plot_normal_distribution(list_to_plot[k][keys[0]], 0, 1)
            plt.title(f"Normal distribution of {plot_titles[k]},\nmean = {list_to_plot[k][key].mean()}, std = {list_to_plot[k][key].std()}")
            plt.grid()
        plt.subplots_adjust(top=1., hspace=0.5)
        plt.suptitle(f'key = {key}', y=0)
        plt.show()