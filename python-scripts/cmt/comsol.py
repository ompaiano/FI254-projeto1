import pandas as pd


def read_csv_from_comsol(path, delimiter=",", header=4, skiprows=0, verbose=False):
    df_raw = pd.read_csv(path, delimiter=delimiter, header=header, skiprows=skiprows)
    if verbose:
        print("df shape: ", df_raw.shape)
    df_raw = df_raw.map(lambda x: complex(x.replace("i", "j")) if isinstance(x, str) else x)
    df_raw.rename(columns={df_raw.columns[0]: 'gap', df_raw.columns[1]: 'neff'}, inplace=True)
    if verbose:
        print(df_raw.head())
    return df_raw