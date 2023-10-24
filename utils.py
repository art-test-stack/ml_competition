import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
import scipy

import settings 

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

def read_files(diff_path: str = ''):
    train_a = pd.read_parquet(diff_path + settings.A.train_targets)
    train_b = pd.read_parquet(diff_path + settings.B.train_targets)
    train_c = pd.read_parquet(diff_path + settings.C.train_targets)

    X_train_estimated_a = pd.read_parquet(diff_path + settings.A.X_train_estimated)
    X_train_estimated_b = pd.read_parquet(diff_path + settings.B.X_train_estimated)
    X_train_estimated_c = pd.read_parquet(diff_path + settings.C.X_train_estimated)

    X_train_observed_a = pd.read_parquet(diff_path + settings.A.X_train_observed)
    X_train_observed_b = pd.read_parquet(diff_path + settings.B.X_train_observed)
    X_train_observed_c = pd.read_parquet(diff_path + settings.C.X_train_observed)

    X_test_estimated_a = pd.read_parquet(diff_path + settings.A.X_test_estimated)
    X_test_estimated_b = pd.read_parquet(diff_path + settings.B.X_test_estimated)
    X_test_estimated_c = pd.read_parquet(diff_path + settings.C.X_test_estimated)

    return train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c


def get_reshaped_files(diff_path: str = ''):
    X_train_estimated_a = pd.read_parquet(diff_path + settings.A_reshaped.X_train_estimated)
    X_train_estimated_b = pd.read_parquet(diff_path + settings.B_reshaped.X_train_estimated)
    X_train_estimated_c = pd.read_parquet(diff_path + settings.C_reshaped.X_train_estimated)

    X_train_observed_a = pd.read_parquet(diff_path + settings.A_reshaped.X_train_observed)
    X_train_observed_b = pd.read_parquet(diff_path + settings.B_reshaped.X_train_observed)
    X_train_observed_c = pd.read_parquet(diff_path + settings.C_reshaped.X_train_observed)

    X_test_estimated_a = pd.read_parquet(diff_path + settings.A_reshaped.X_test_estimated)
    X_test_estimated_b = pd.read_parquet(diff_path + settings.B_reshaped.X_test_estimated)
    X_test_estimated_c = pd.read_parquet(diff_path + settings.C_reshaped.X_test_estimated)

    return X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c


def get_days_to_predict(dframe = None, diff_path: str='../', date_key = 'date_forecast'):
    dframe = pd.read_parquet(diff_path + settings.A.X_test_estimated) if dframe is None else dframe
    days = []
    for k in range(1, 31):
        k0  = f'0{k}' if k < 10 else str(k)
        k01 = f'0{k + 1}' if k < 9 else str(k + 1)
        if np.array(dframe[(dframe[date_key] > f'2023-05-{k0}') & (dframe[date_key] < f'2023-05-{k01}')]).shape[0] != 0:
            days.append(f'2023-05-{k0}')
    if np.array(dframe[(dframe[date_key] > f'2023-05-31') & (dframe[date_key] < f'2023-06-01')]).shape[0] != 0: days.append('2023-05-31')
    for k in range(1, 30):
        k0  = f'0{k}' if k < 10 else str(k)
        k01 = f'0{k + 1}' if k < 9 else str(k + 1)
        if np.array(dframe[(dframe[date_key] > f'2023-06-{k0}') & (dframe[date_key] < f'2023-06-{k01}')]).shape[0] != 0:
            days.append(f'2023-06-{k0}')
    if np.array(dframe[(dframe[date_key] > f'2023-06-30') & (dframe[date_key] < f'2023-07-01')]).shape[0] != 0: days.append('2023-06-30')
    for k in range(1, 15):
        k0  = f'0{k}' if k < 10 else str(k)
        k01 = f'0{k + 1}' if k < 9 else str(k + 1)
        if np.array(dframe[(dframe[date_key] > f'2023-07-{k0}') & (dframe[date_key] < f'2023-07-{k01}')]).shape[0] != 0:
            days.append(f'2023-07-{k0}')
    return days


def build_corr_matrix(data_frames, figsize=(20,20), annot=True):
    plt.figure(figsize=figsize)
    sns.heatmap(data_frames.corr(method='pearson'), annot=annot, cmap=plt.cm.Reds)

def concat_frames(frames: list = [], keys: list = [], on: str = ''):
    frames_concat = pd.concat(frames, keys=keys)
    return frames_concat.reset_index(level=0, inplace=True, names=on)

def reshape_frame_to_match_prediction_format(frame, directory):
    groups = [frame[i:i+4] for i in range(0, len(frame), 4)]
    groups_agg = []
    for groupe in tqdm(groups):
        groups_without_nan = groupe.fillna('')
        new_input = groups_without_nan.stack().reset_index(drop=True)
        groups_agg.append(new_input)

    new_frame = pd.concat(groups_agg, axis=1, ignore_index=False).T
    columns = []
    for k in range(4):
        [columns.append(f"{c}_{k}") for c in frame.keys()]

    new_frame.columns = columns
    new_frame.reset_index(drop=True, inplace=True)
    new_frame.replace('', np.nan).to_parquet(directory)
    return new_frame

def get_most_important_keys():
    return ['date_forecast', 'clear_sky_energy_1h:J', 'clear_sky_rad:W', 'diffuse_rad:W', 'diffuse_rad_1h:J','direct_rad:W']

# def convert_date_to_index()