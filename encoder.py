from settings import *
import math
import numpy as np
from PIL import Image as im


class Encoder:
    def __init__(self, df):
        self.length_of_packet = len(df.columns)
        self.picture_pixels_number = 1024
        self.number_of_packets_in_picture = math.floor(self.picture_pixels_number / self.length_of_packet)

    def start_encoding_process(self, df, path=""):
        df = df.drop(columns=COLUMNS_TO_DROP_BEFORE_ENCODING)
        encoded_session = self.encode_session(df)
        if SAVE_ENCODED_PICTURES and path:
            data = im.fromarray(encoded_session)
            self.save_encoded_picture(data, path)
        return encoded_session


    @staticmethod
    def save_encoded_picture(data, path):
        data.save(path)

    def encode_session(self, df):
        used_logs = df.head(self.number_of_packets_in_picture)
        val = list(used_logs.values)
        result = [v for sub in val for v in sub]
        missing = max(self.picture_pixels_number - len(result), 0)
        padded = [0] * missing
        used_logs = result[:1025] + padded
        if len(used_logs) != self.picture_pixels_number:
            padded = [0] * (self.picture_pixels_number - len(used_logs))
            used_logs = used_logs + padded
        used_logs = [float(val) for val in used_logs]
        encoded_session = np.array(used_logs)
        try:
            encoded_session = np.reshape(encoded_session, (32, 32))
        except ValueError:
            return
        encoded_session = encoded_session * 255
        encoded_session = encoded_session.astype(np.uint8)
        return encoded_session
