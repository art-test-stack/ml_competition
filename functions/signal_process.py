import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime, timedelta

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

import sys
sys.path.append('../')
import functions.utils as utils

# USEFUL VALUES

# start_date0 = { 'a': '2020-10-21', 'b': '2020-03-15', 'c': '2020-04-01' }
# end_date0 = { 'a': '2022-10-21', 'b': '2022-03-15', 'c': '2022-04-01'}


start_date0 = { 
    'a': datetime(2020,10,21, 0, 0, 0), 
    'b': datetime(2020, 3, 15, 0, 0, 0), 
    'c': datetime(2020, 4, 1, 0, 0, 0) 
}

# start_date2 = { 
#     'a': datetime(2019, 6, 2), 
#     'b': datetime(2018, 12, 31), 
#     'c': datetime(2019, 9, 4) 
# }

end_date0 = { 
    'a': datetime(2022, 10, 21, 0, 0, 0), 
    'b': datetime(2022, 3, 15, 0, 0, 0), 
    'c': datetime(2022, 4, 1, 0, 0, 0)
}
split_date0 = {
    'a': pd.to_datetime("2022-07-21"),
    'b': pd.to_datetime("2021-12-15"),
    'c': pd.to_datetime("2022-01-01")
}

locations = ['a', 'b', 'c']
hours = [ f"0{h}" if h < 10 else str(h) for h in range(24) ]

# UTILITY FUNCTIONS


def get_model_freq_amp_phase(fft_values, threshold=60, sample_rate=1):

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

def get_dates_index(dates1, dates2):
    pass
# IMPORTANT FUNCTION:
# See example on:
# signal_analysis_example.ipynb

def get_location_key(loc):
    return [ 3 * k for k in range(4) ] if loc =='a' else [ 3 * k + 1 for k in range(4) ] if loc =='b' else [ 3 * k + 2 for k in range(4) ] if loc =='c' else None

class SignalProcess:
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

    def __init__(
            self,
            location='a',
            diff_path='../',
            normalization='mean_std',
            sample_rate=1,
            nb_frequencies=1, 
            initial_threshold=0, 
            filter_step=.1,
            start_date=None,
            end_date=None,
            split_date=None
            ):
        self.location = location
        train, self.X_tr_est, self.X_tr_obs, self.X_te_est = [utils.read_files(diff_path=diff_path)[i] for i in get_location_key(location)]
        
        self.diff_path = diff_path
        train = train.rename(columns={'pv_measurement': 'y'})
        train = train.dropna(subset='y')
        train = pd.merge(self.X_tr_obs.rename(columns={'date_forecast': 'time'}), train, on='time', how='inner')#.drop(columns=self.X_tr_obs.rename(columns={'date_forecast': 'time'}).columns)
        columns_to_drop = [ c for c in self.X_tr_obs.columns if c != 'date_forecast']
        self.train = train.drop(columns=columns_to_drop)
        
        self.train = train[(train['time'] >= start_date) & (train['time'] < end_date)]

        self.start_date = start_date0[location] if start_date is None else start_date
        self.end_date = end_date0[location]if end_date is None else end_date
        self.split_date = split_date0[location]if split_date is None else split_date

        self.days_predicted = 0


        self.train_dates = self.train['time']
        self.sample_rate = sample_rate
        self.mean_y = self.train.mean()
        self.std_y = self.train.std()
        
        self.preprocess_data(normalization)
        self.filter_frequencies(nb_frequencies, initial_threshold, filter_step, sample_rate)

    def split_signal_in_hours(self):
        Y_train, Y_test, Y_total = {}, {}, {} 
        for h in hours: 
            Y = self.train[self.train['time'].dt.strftime('%H:%M:%S').str.endswith(f'{h}:00:00')]
            Y_train[h] = np.array(Y[Y['time'] < self.split_date]['y'])
            Y_test[h] = np.array(Y[Y['time'] >= self.split_date]['y']) 
            Y_total[h] =  np.array(Y['y']) 
        return Y_train, Y_test, Y_total
    
    def store_means_and_stds_by_hours(self, signal):
        self.mean_on_h = { h: np.mean(signal[h]) for h in hours }
        self.std_on_h = { h: np.std(signal[h]) for h in hours }

    def store_min_max_by_hours(self, signal):
        self.min = { h: np.min(signal[h]) for h in hours }
        self.max = { h: np.max(signal[h]) for h in hours }

    def normalize_by_hours_mean_std(self, signal_by_hours):
        not_std = [ 0, .0, float('inf'), float('-inf'), float('nan') ]

        return { h: (signal_by_hours[h] - self.mean_on_h[h])/ self.std_on_h[h] if self.std_on_h[h] not in not_std else (signal_by_hours[h] - self.mean_on_h[h]) for h in hours }
        
    def normalize_by_hours_min_max(self, signal_by_hours):
        return { h: (signal_by_hours[h] - self.min[h])/ (self.max[h] - self.min[h]) for h in hours}
            
    def preprocess_data(self, norm = 'mean_std'):
        Y_train, Y_test, Y_total = self.split_signal_in_hours()

        self.store_means_and_stds_by_hours(Y_total) if norm == 'mean_std' else self.store_min_max_by_hours(Y_total) if norm == 'min_max' else None
        self.train_normalized = self.normalize_by_hours_mean_std(Y_train) if norm == 'mean_std' else self.normalize_by_hours_min_max(Y_train) if norm == 'min_max' else None
        self.test_normalized = self.normalize_by_hours_mean_std(Y_test) if norm == 'mean_std' else self.normalize_by_hours_min_max(Y_test) if norm == 'min_max' else None
        self.total_normalized = self.normalize_by_hours_mean_std(Y_total) if norm == 'mean_std' else self.normalize_by_hours_min_max(Y_total) if norm == 'min_max' else None

    def filter_frequencies(self, nb_frequencies=1, initial_threshold=0, filter_step=.1, sample_rate=1):
        self.thresholds = { h: get_thresholds_to_get_n_freq(signal=self.total_normalized[h], nb_freq=nb_frequencies, threshold=initial_threshold, step=filter_step) for h in hours }
        
        self.model = { h: get_model_freq_amp_phase(fft_values=np.fft.fft(self.total_normalized[h]), threshold=self.thresholds[h], sample_rate=sample_rate) for h in hours }

    def reconstruct_normed_filtred_signal(self, nb_days_to_predict):
        self.days_predicted += nb_days_to_predict
        y_pred = {}

        hour = 0

        for h in hours: 

            all_days = []
            date_to_append = self.start_date

            date_format = '%Y-%m-%d'

            last_date = self.end_date

            while date_to_append <= last_date:
                all_days.append(date_to_append.strftime(date_format))
                date_to_append += timedelta(days=1)

            y_pred_raw = reconstruct_signal(self.model[h], duration=len(all_days) + nb_days_to_predict, sample_rate=self.sample_rate)
            
            # y_pred_raw = (y_pred_raw - np.mean(y_pred_raw)) / np.std(y_pred_raw) if np.std(y_pred_raw) != 0. else (y_pred_raw - np.mean(y_pred_raw)) / np.std(y_pred_raw)
            # y_pred_raw = (y_pred_raw - np.mean(y_pred_raw)) / np.std(y_pred_raw) if np.std(y_pred_raw) != 0. else (y_pred_raw - np.mean(y_pred_raw)) / np.std(y_pred_raw)
            
            y_pred_scaled = np.where(
                np.real(y_pred_raw) > np.min(self.total_normalized[h]),
                np.real(y_pred_raw),
                np.min(np.min(self.total_normalized[h])))
            
            y_pred_normalized = y_pred_scaled # - np.mean(y_pred_scaled) self.std_on_h[h] + self.mean_on_h[h]
            y_pred[h] = y_pred_normalized # * self.mean_on_h[h] / np.mean(y_pred_normalized) if np.mean(y_pred_normalized) != 0 else np.zeros(shape=y_pred_normalized.shape)
            hour += 1

        self.y_pred = y_pred
        return y_pred
        
    def unnorm_signal(self, signal):
        return { h: signal[h] * self.std_on_h[h] + self.mean_on_h[h] for h in hours }

    def convert_hours_to_days(self, signal):
        min_len = np.min([ len(signal[h]) for h in hours ])
        y_pred = []
        for d in range(min_len):
            for h in hours:
                y_pred.append(signal[h][d])
        return np.array(y_pred)

    def convert_to_df(self, signal):
        dates = [(self.train_dates + d).strftime("%Y-%m-%d") for d in [ timedelta(days=k) for k in range(self.days_predicted)]]
        df = pd.DataFrame(columns=["time", "pv_measurement"])
        for date_i in range(len(dates)):
            for h in hours:
                df[(f"{dates[date_i]} {h}:00:00").strftime("%Y-%m-%d HH:MM:SS")] = self.unnorm_signal[h][date_i] * self.std_y + self.mean_y
        return df
    
    def get_filtered_signal_on_prediction_dates(self):
        days_to_predict = utils.get_days_to_predict(diff_path=self.diff_path)
        
        all_days = []
        date_to_append = self.start_date

        date_format = '%Y-%m-%d'

        last_date = datetime.strptime(max(days_to_predict), date_format)

        while date_to_append <= last_date:
            all_days.append(date_to_append.strftime(date_format))
            date_to_append += timedelta(days=1)

        index_to_predict = [ all_days.index(d) for d in days_to_predict ]

        nb_days_to_predict = ( last_date - self.end_date ).days
        
        reconstructed_signal = self.reconstruct_normed_filtred_signal(nb_days_to_predict)

        filtered_signal_on_prediction_dates = { h: [np.real(reconstructed_signal[h][i]) for i in index_to_predict] for h in hours }
        return filtered_signal_on_prediction_dates

    def get_filtered_signal_on_training_dates(self):
        
        date_format = '%Y-%m-%d'
        index_to_predict = { h: [] for h in hours }
        training_dates = { h: [] for h in hours }
        all_days = { h: [] for h in hours }

        for h in hours:
    
            date_to_append = self.start_date + timedelta(hours=hours.index(h))
            last_date = self.end_date

            while date_to_append <= last_date:
                all_days[h].append(date_to_append.strftime(date_format))
                date_to_append += timedelta(days=1)
            training_dates[h] = self.train_dates[self.train_dates.dt.hour == hours.index(h)]
            index_to_predict[h] = [ all_days[h].index(d) for d in training_dates[h].dt.strftime(date_format) ]

        reconstructed_signal = self.reconstruct_normed_filtred_signal(0)

        filtered_signal_on_training_dates = { h: [np.real(reconstructed_signal[h][i]) for i in index_to_predict[h]] for h in hours }
        return filtered_signal_on_training_dates


    def setup_pipeline(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X_train, X_test):
        pass