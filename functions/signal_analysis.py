import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy 

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

import sys
sys.path.append('../')
import functions.utils as utils
import settings

# USEFUL VALUES

start_date0 = { 'a': '2020-10-21', 'b': '2020-03-15', 'c': '2020-04-01' }
end_date0 = { 'a': '2022-10-21', 'b': '2022-03-15', 'c': '2022-04-01'}

locations = ['a', 'b', 'c']
hours = [ f"0{h}" if h < 10 else str(h) for h in range(24) ]

# UTILITY FUNCTIONS


def get_model(fft_values, threshold=60, sample_rate=1):

    n = len(fft_values)

    frequencies = np.fft.fftfreq(n, 1 / sample_rate)
    amplitudes = fft_values * (np.abs(fft_values) > threshold)
    phases = np.angle(fft_values)
    return {"frequencies": frequencies, "amplitudes": amplitudes, "phases": phases}


def reconstruct_signal(model, duration, sample_rate = 1):
    frequencies = model["frequencies"]
    amplitudes = model["amplitudes"]
    phases = model["phases"]
    
    t = np.arange(0, duration, 1 / sample_rate)
    signal = np.zeros(len(t), dtype=np.complex128)
    
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal += amp * np.exp(2j * np.pi * freq * t + phase)
    
    return signal / len(frequencies)


def get_thresholds_to_get_n_freq(signal, nb_freq, threshold, step):
    assert step > 0
    fft = np.fft.fft(signal)
    abs_fft = np.abs(fft[:len(fft)//2])

    freqs = [ f for f in abs_fft if f > threshold ]
    threshold += step
    while len(freqs) > nb_freq:
        freqs = [ f for f in abs_fft if f > threshold ]
        threshold += step
    threshold = threshold if len(freqs) > 0 else threshold - step
    return threshold


# IMPORTANT FUNCTION:
# See example on:
# signal_analysis_example.ipynb


def get_normalized_y_and_pred_separated_by_hours_and_location(
        diff_path='../',
        start_date=start_date0,
        end_date=end_date0,
        nb_frequences=2,
        nb_days_to_predict=0,
        sample_rate=1, # 1 by day
        factor_to_fit=1
    ):
    """
    GET y normalized and y_pred from signal analysis

    input:
        - diff_path (str): path to access A, B and C data
        - start_date (dict): { loc: start_date } start date of the analysis
        - end_date (dict): { loc: end_date } end date of the analysis
        - nb_frequences (int): nb of frequences we want to keep in the filter
        - nb_days_to_predict: number of days we want to extend the prediction

    output:
        - Y_train (dict): Y normalized {loc: { h: y_value[loc][h] }}
        - Y_pred (dict): Y pred normalized {loc: { h: y_pred_value[loc][h] }}
    """

    train_a, train_b, train_c, _, _, _, _, _, _, _, _, _ = utils.read_files(diff_path=diff_path)

    trains = [ train_a, train_b, train_c ]

    train_ = {}
    train_ = { locations[k]: trains[k].rename(columns={'time': 'ds', 'pv_measurement': 'y'}) for k in range(len(trains))}
    train_ = { loc: train_[loc][(train_[loc]["ds"] < end_date[loc]) & (train_[loc]["ds"] > start_date[loc])] for loc in locations }

    mean_y_ = { loc: train_[loc]["y"].mean() for loc in locations }
    std_y_ = { loc: train_[loc]["y"].std() for loc in locations }

    _Y_train_ = { loc: { h: train_[loc][train_[loc]['ds'].dt.strftime('%H:%M:%S').str.endswith(f'{h}:00:00')] for h in hours } for loc in locations }
    
    _Y_train_['b'] = { h: _Y_train_['b'][h].dropna(subset="y") for h in hours }
    _Y_train_['c'] = { h: _Y_train_['c'][h].dropna(subset="y") for h in hours }

    not_std = [ 0, .0, float('inf'), float('-inf'), float('nan') ]
    Y_train = { loc: 
            { h: ( _Y_train_[loc][h]['y'] - mean_y_[loc] ) / std_y_[loc] # y_std_[loc][h] 
                if std_y_[loc] not in not_std 
                else _Y_train_[loc][h] - mean_y_[loc] for h in hours 
                } for loc in locations 
            }

    Y_train = { loc: 
            { h: np.array( _Y_train_[loc][h]['y'] - np.min(_Y_train_[loc][h]['y']) ) / ( np.max(_Y_train_[loc][h]['y']) - np.min(_Y_train_[loc][h]['y'])) # y_std_[loc][h] 
                if np.max(_Y_train_[loc][h]['y']) - np.min(_Y_train_[loc][h]['y']) != .0
                else _Y_train_[loc][h]['y'] for h in hours 
                } for loc in locations 
            }

    # DROP NA FOR C (important for c fft)
    # Y_train['c'] = { h: Y_train['c'][h].dropna() for h in hours }

    thresholds = { loc: { h: get_thresholds_to_get_n_freq(signal=Y_train[loc][h], nb_freq=nb_frequences, threshold=0, step=.5) for h in hours } for loc in locations }

    model = { loc: { h: get_model(fft_values=np.fft.fft(Y_train[loc][h]), threshold=thresholds[loc][h], sample_rate=sample_rate) for h in hours } for loc in locations }
    pred_from_model_data = { loc: { h: reconstruct_signal(model[loc][h], duration=len(model[loc][h]["frequencies"]) + nb_days_to_predict, sample_rate=sample_rate) for h in hours } for loc in locations }

    y_filtred_fit = { loc: { h: (pred_from_model_data[loc][h] - np.mean(pred_from_model_data[loc][h])) / np.std(pred_from_model_data[loc][h]) for h in hours } for loc in locations }
    
    factor_to_fit = 1
    y_filtred_fit = { 
        loc: {
            h: np.where(
                np.real(y_filtred_fit[loc][h]) > np.min(np.array(Y_train[loc][h])),
                np.real(y_filtred_fit[loc][h]) / factor_to_fit,
                np.min(np.array(Y_train[loc][h]))) for h in hours 
            } for loc in locations 
        }
    y_filtred_fit = { loc: { h: pred_from_model_data[loc][h] * np.std(Y_train[loc][h]) + np.mean(Y_train[loc][h]) for h in hours } for loc in locations }
    y_filtred_fit = { 
        loc: {
            h: np.where(
                np.real(y_filtred_fit[loc][h]) > np.min(np.array(Y_train[loc][h])),
                np.real(y_filtred_fit[loc][h]) / factor_to_fit,
                np.min(np.array(Y_train[loc][h]))) for h in hours 
            } for loc in locations 
        }
    return Y_train, y_filtred_fit


def format_signal_to_final_format(
        y_to_format
    ):

    train_a, train_b, train_c, _, _, _, _, _, _, _, _, _ = utils.read_files(diff_path=diff_path)
    

def filter_dates_when_constants(df, date_c = 'time', y = 'pv_measurement', delta = { 'days': 3 }):
    df = df.copy()
    mask_y_change = df[y] != df[y].shift(1)

    start_date = None
    end_date = None

    constant_periods = []

    for index, row in df.iterrows():
        if not mask_y_change[index]:
            if start_date is None:
                start_date = row[date_c]
            end_date = row[date_c]
        else:
            if start_date is not None and (end_date - start_date) >= pd.Timedelta(**delta):
                constant_periods.append((start_date, end_date))
            start_date = None
            end_date = None

    if start_date is not None and (end_date - start_date) >= pd.Timedelta(**delta):
        constant_periods.append((start_date, end_date))
    return constant_periods


def delete_date_range_from_df(df, dates, date_c = 'time'):
    df = df.copy()
    c = 0
    for start_date, end_date in dates:
        mask = (df[date_c] >= start_date) & (df[date_c] < end_date)
        df = df[~mask]
    df.reset_index(drop=True, inplace=True)
    return df


def get_fft_transforms(train):
    y = train["pv_measurement"].dropna().values
    time_diff = train["time"].diff().mean().total_seconds()
    sampling_rate = 1 / time_diff

    n = len(y)
    freq = np.fft.fftfreq(n, 1 / sampling_rate)
    fft_y = np.fft.fft(y)
    amp_fft_y = np.abs(fft_y)
    phase = np.angle(fft_y)
    return freq, fft_y, amp_fft_y, phase, sampling_rate


def print_peak_frequencies(data, amp_fft_y, freq, threshold, loc):
    peaks, _ = scipy.signal.find_peaks(amp_fft_y[:len(amp_fft_y)//2], height=threshold)
    peak_frequencies = freq[:len(freq)//2][peaks]

    period_size = int(1/peak_frequencies[0])
    continuous_component = np.mean(data[loc]["pv_measurement"].dropna().values[:period_size])

    print("Location:", loc)
    print(f'Most important periods (in days): \n{1 / peak_frequencies / 3600 / 24}')
    print(f'Value of the continous component: {continuous_component}\n\n')


def get_filtred_signal(signal, nb_freqs, sample_rate, nb_days_to_predict = 0, threshold = 0, scaler = StandardScaler):
    scaler_pred = scaler
    scaler = scaler()
    Y_normed = scaler.fit_transform(np.array(signal['pv_measurement'].dropna()).reshape(-1, 1)).reshape(-1)

    threshold = get_thresholds_to_get_n_freq(signal=Y_normed, nb_freq=nb_freqs, threshold=0, step=.5)
    model = get_model(fft_values=np.fft.fft(Y_normed), threshold=threshold, sample_rate=sample_rate)
    pred_from_model_data = np.real(reconstruct_signal(model, duration=len(model["frequencies"]) + nb_days_to_predict, sample_rate=sample_rate)) 
    scaler_pred = scaler_pred()
    pred_normed = scaler_pred.fit_transform(pred_from_model_data.reshape(-1, 1)).reshape(-1)

    Y_filtred = scaler.inverse_transform(pred_normed.reshape(-1, 1)).reshape(-1)

    # If we want to filter negative values
    Y_filtred[Y_filtred < 0] = 0
    return Y_filtred