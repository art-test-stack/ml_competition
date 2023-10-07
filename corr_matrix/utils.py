import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

def read_files():
    train_a = pd.read_parquet('../A/train_targets.parquet')
    train_b = pd.read_parquet('../B/train_targets.parquet')
    train_c = pd.read_parquet('../C/train_targets.parquet')

    X_train_estimated_a = pd.read_parquet('../A/X_train_estimated.parquet')
    X_train_estimated_b = pd.read_parquet('../B/X_train_estimated.parquet')
    X_train_estimated_c = pd.read_parquet('../C/X_train_estimated.parquet')

    X_train_observed_a = pd.read_parquet('../A/X_train_observed.parquet')
    X_train_observed_b = pd.read_parquet('../B/X_train_observed.parquet')
    X_train_observed_c = pd.read_parquet('../C/X_train_observed.parquet')

    X_test_estimated_a = pd.read_parquet('../A/X_test_estimated.parquet')
    X_test_estimated_b = pd.read_parquet('../B/X_test_estimated.parquet')
    X_test_estimated_c = pd.read_parquet('../C/X_test_estimated.parquet')

    return train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c

def build_corr_matrix(data_frames, figsize=(20,20), annot=True):
    plt.figure(figsize=figsize)
    sns.heatmap(data_frames.corr(method='pearson'), annot=annot, cmap=plt.cm.Reds)