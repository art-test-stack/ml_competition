

def normalize_df(df):
    df_normalized = df.copy()
    for c in df.columns :
        if (c not in ['date_forecast','time']):
            df_normalized[c] = (df_normalized[c] - df_normalized[c].mean()) / df_normalized[c].std() if df_normalized[c].std() != 0 else (df_normalized[c] - df_normalized[c].mean())

    return df_normalized  