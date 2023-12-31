# Rapport LaTeX: 

[Lien du rapport](https://www.overleaf.com/2942188165ppvhmyjptyxq#ea4a8b)

# Google Doc file:

[Lien du Google doc](https://docs.google.com/document/d/1HqctPEYCdJXFtMEsj8JrUAaVhz6qA8VkSL6zdqUSfrs/edit?usp=sharing)

# Google colab file (Valentin):

[Lien du Google colab de test](https://colab.research.google.com/drive/11yQYJJiiXjr2ZB4mD-PZoiRMpRZ6ajNQ?usp=sharing)

[Lien du Google colab dépot Kaggle](https://colab.research.google.com/drive/1eLPQoJ8eKEKOLXDn7_LADMFoXpte3anO?usp=sharing)

[Lien du Google colab étude sans TS](https://colab.research.google.com/drive/1bMi9gyoiIkMLv2NDGDHLrlxDz2Ykbf5e#scrollTo=4nl1rTLkQgsv)

[Lien du Google colab analyse heure par heure](https://colab.research.google.com/drive/1koqf9g9t5JexdVAuAvjwyCN8CiMBCkki?usp=sharing)

[Lien du Google colab XGBoost 170](https://colab.research.google.com/drive/16FTpI4JkbT4s9JRzNp7lxiy6aJ5I89PK?usp=sharing)

[Lien du Google colab Rapport partie Valentin]
(https://colab.research.google.com/drive/1GmWXBJ6KZzWMR-UWPnqAsfYB9izNmwbH?usp=sharing)

[Lien du Google colab Rapport partie Nahel]
(https://colab.research.google.com/drive/1JBv_KbkKGDA_AAoVTtvLLhVMST5xFo3M?usp=sharing)

# Solar Datahead Forecast Data

This dataset provides data for evaluating solar production dayahead forecasting methods.
The dataset contains three locations (A, B, C), corresponding to office buildings with solar panels installed.
There is one folder for each location.

There are 4 files in each folder:

1. train_targets.parquet - target values for the train period (solar energy production)
2. X_train_observed.parquet - actual weather features for the first part of the training period
2. X_train_estimated.parquet - predicted weather features for the remaining part of the training period
2. X_test_estimated.parquet - predicted weather features for the test period

For Kaggle submissions we have two more files: 
1. test.csv — test file with zero predictions (for all three locations)
2. sample_submission_kaggle.csv — sample submission in the Kaggle format (for all three locations)

Kaggle expects you to submit your solutions in the "sample_sumbission_kaggle.csv" format. Namely, you need to have two columns: "id" and "prediction".
The correspondence between id and time/location is in the test.csv file. An example solution is provided in "read_files.ipynb"

All files that are in the parquet format that can be read with pandas:
```shell
pd.read_parquet()
```

Baseline and targets production values have hourly resolution.
Weather has 15 min resolution.
Weather parameter descriptions can be found [here](https://www.meteomatics.com/en/api/available-parameters/alphabetic-list/).

There is a distinction between train and test features.
For training, we have both observed weather data and its forecasts, while for testing we only have forecasts.
While file `X_train_observed.parquet` contains one time-related column `date_forecast` to indicate when the values for the current row apply,
both `X_train_estimated.parquet` and  `X_test_estimated.parquet` additionally contain `date_calc` to indicate when the forecast was produced.
This type of test data makes evaluation closer to how the forecasting methods that are used in production.
Evaluation measure is [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error).
