import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

import sys
sys.path.append('../')
import utils
import settings


start_date0 = { 'a': '2020-10-21', 'b': '2020-03-15', 'c': '2020-04-01' }
end_date0 = { 'a': '2022-10-21', 'b': '2022-03-15', 'c': '2022-04-01'}

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
        sample_rate=1 # 1 by day
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

    locations = ['a', 'b', 'c']
    hours = [ f"0{h}" if h < 10 else str(h) for h in range(24) ]
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

    # Factor to fit Y_train middle (set max to .5 if factor_to_fit == 2)
    factor_to_fit = 1
    y_filtred_fit = { 
        loc: {
            h: np.where(
                np.real(y_filtred_fit[loc][h]) > np.min(np.array(Y_train[loc][h])),
                np.real(y_filtred_fit[loc][h]) / factor_to_fit,
                np.min(np.array(Y_train[loc][h]))) for h in hours 
            } for loc in locations 
        }

    return Y_train, y_filtred_fit