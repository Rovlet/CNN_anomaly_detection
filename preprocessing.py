import os
import sys
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing

from encoder import Encoder
from settings import *


def prepare_data(df):
    df = pd.get_dummies(df, columns=DUMMIES_COLUMNS, dtype=float)
    df.replace([np.inf, -np.inf], sys.float_info.max, inplace=True)
    df.fillna(0.0)
    df = df.drop(columns=COLUMNS_TO_DROP)

    columns = df.columns
    columns = [c for c in columns if c not in COLUMNS_TO_IGNORE]

    min_max_scaler = preprocessing.MinMaxScaler()
    df[[f"{column}" for column in columns]] = min_max_scaler.fit_transform(df[[f"{column}" for column in columns]])
    return df


def prepare_session_data(df):
    start_time = df[' Timestamp'].iloc[0]
    df.loc[df[' Timestamp'] != start_time, ' Timestamp'] = 0
    df.loc[df[' Timestamp'] == start_time, ' Timestamp'] = 1
    return df


def drop_highly_correlated_columns(df):
    corrMatrix = df.corr().abs()
    upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(to_drop, axis=1, inplace=True)

    if DISPLAY_CORR_MATRICES:
        sn.heatmap(corrMatrix)
        plt.show()
    return df


if __name__ == '__main__':
    directory = os.listdir(DATA_PATH)
    files = [filename for filename in directory if filename.endswith('csv')]
    frames = []
    for file in files:
        df = pd.read_csv(DATA_PATH + file)
        encoder = Encoder(df)
        data_file = file.strip('.csv')
        folder = os.path.join(f'./data/{data_file}')
        if not os.path.isdir(folder):
            os.makedirs(folder)

        df = prepare_data(df)
        df = drop_highly_correlated_columns(df)
        all_unique_sessions = list(df['Flow ID'].unique())

        for i, session in enumerate(all_unique_sessions):
            session_df = df.loc[df['Flow ID'] == session]
            if len(session_df) >= MINIMUM_LOGS_FROM_SESSION:
                session_df = prepare_session_data(session_df)
                frames.append(session_df)
                encoder.start_encoding_process(df, path=f'./data/{data_file}/{i}.png')

    result_df = pd.concat(frames)
    result_df.to_csv(f"results.csv")
