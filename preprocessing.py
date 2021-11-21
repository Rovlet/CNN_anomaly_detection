import os
import sys
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import multiprocessing
import json
from PIL import Image as im

from sklearn.model_selection import train_test_split

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
    encoder = Encoder(df)
    if df.empty:
        return
    start_time = df[' Timestamp'].iloc[0]
    df.loc[df[' Timestamp'] != start_time, ' Timestamp'] = 0
    df.loc[df[' Timestamp'] == start_time, ' Timestamp'] = 1
    label = df[' Label'].iloc[0]
    encoded_session = encoder.start_encoding_process(df)
    return df, encoded_session, label


def get_highly_correlated_columns(df):
    corrMatrix = df.corr().abs()
    upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    if DISPLAY_CORR_MATRICES:
        sn.heatmap(corrMatrix)
        plt.show()
        corrMatrix = df.corr().abs()
        sn.heatmap(corrMatrix)
        plt.show()
    return to_drop


def get_rows_with_the_same_id(arguments):
    df, session_id = arguments
    return df.loc[df['Flow ID'] == session_id]


if __name__ == '__main__':
    directory = os.listdir(DATA_PATH)
    files = [filename for filename in directory if filename.endswith('csv')]
    frames = []
    for file in files:
        df = pd.read_csv(DATA_PATH + file, encoding='utf-8',
                         dtype={'Flow ID': np.unicode_, ' Source IP': np.unicode_, " Destination IP": np.unicode_,
                                " Timestamp": np.unicode_, " Label": np.unicode_})
        frames.append(df)
    df = pd.concat(frames)

    df = prepare_data(df)
    to_drop = get_highly_correlated_columns(df)

    df.drop(to_drop, axis=1, inplace=True)

    all_unique_sessions = list(df['Flow ID'].unique())

    num_workers = multiprocessing.cpu_count() - 3
    print(f"Numbers of workers: {num_workers}")
    arguments = [[df, session_id] for session_id in all_unique_sessions]

    with multiprocessing.Pool(num_workers) as pool:
        frames = pool.map(get_rows_with_the_same_id, arguments)
    arguments = frames


    X = []
    y = []

    with multiprocessing.Pool(num_workers) as pool:
        frames = pool.map(prepare_session_data, arguments)

    all_df = []
    for i, frame in enumerate(frames):
        if frame is not None:
            df, encoded_session, label = frame
            if encoded_session is not None:
                all_df.append(df)
                X.append(encoded_session)
                y.append(label)
                if SAVE_ENCODED_PICTURES:
                    data = im.fromarray(encoded_session)
                    Encoder.save_encoded_picture(data, path=f'./pictures/{i}.png')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    np.save('X_train.npy', X_train, allow_pickle=True)
    np.save('X_test.npy', X_test, allow_pickle=True)
    np.save('y_train.npy', y_train, allow_pickle=True)
    np.save('y_test.npy', y_test, allow_pickle=True)

    all_df = pd.concat(all_df)
    all_df.to_csv(f'results.csv', header=True, index=False)
