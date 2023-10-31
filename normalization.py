import numpy as np

def normalize_df(df):
    df_normalized = df.copy()
    for c in df.columns :
        if (c not in ['date_forecast','time']):
            df_normalized[c] = (df_normalized[c] - df_normalized[c].mean()) / df_normalized[c].std() if df_normalized[c].std() != 0 else (df_normalized[c] - df_normalized[c].mean())

    return df_normalized  

def normalize_list_df(dfs):
    dfs_normed = []
    means = []
    stds = []
    k = 0
    for df in dfs:
        means.append([])
        stds.append([])
        for c in df.columns:
            means[k].append(df[c].mean())
            stds[k].append(df[c].std())
        dfs_normed.append(normalize_df(df))
        k += 1
    return dfs_normed, means, stds

def unnorm_df(normed_df, mean, std):
    normed_df = np.array(normed_df)
    print(np.min(normed_df))
    df = np.zeros(shape=normed_df.shape)
    for row in range(len(normed_df)):
        cond = ((normed_df[row] * std + mean) < 11)
        df[row] = 0.0 if cond else (normed_df[row]) * std + mean
        # if (normed_df[row] * std + mean) < 12:
        #     print('here')
        #     # print((normed_df[row] - np.min(normed_df)) * std + mean)
        #     df[row] = 0.0
        # # elif ((normed_df[row] - np.min(normed_df)) * std + mean ):
        #     # print((normed_df[row] - np.min(normed_df)) * std + mean)
        # else:
        #     df[row] = (normed_df[row]) * std + mean
        # df[row] = (normed_df[row] - np.min(normed_df[row])) * std 
        # df[row] += mean
        # df[row] -= np.min(normed_df[row]) * std
    return df