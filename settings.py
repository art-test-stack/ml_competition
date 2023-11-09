class DataFolder:
    def __init__(self, folder_name: str):
        self.folder_name: str
        self.X_test_estimated: str = f"{folder_name}/X_test_estimated.parquet"
        self.X_train_estimated: str = f"{folder_name}/X_train_estimated.parquet"
        self.X_train_observed: str = f"{folder_name}/X_train_observed.parquet"
        self.train_targets: str | None = f"{folder_name}/train_targets.parquet"

A = DataFolder(folder_name='A')
B = DataFolder(folder_name='B')
C = DataFolder(folder_name='C')

A_reshaped = DataFolder(folder_name='A_reshaped')
B_reshaped = DataFolder(folder_name='B_reshaped')
C_reshaped = DataFolder(folder_name='C_reshaped')

A_reshaped3 = DataFolder(folder_name='A_reshaped3')
B_reshaped3 = DataFolder(folder_name='B_reshaped3')
C_reshaped3 = DataFolder(folder_name='C_reshaped3')

utils = 'utils.py'
