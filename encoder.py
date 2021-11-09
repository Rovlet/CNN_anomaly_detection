from settings import *
import math
import numpy as np
from PIL import Image as im


class Encoder:
    def __init__(self, df):
        self.length_of_packet = len(df.columns)
        self.picture_pixels_number = self.length_of_packet * 20
        self.number_of_packets_in_picture = math.floor(self.picture_pixels_number / self.length_of_packet)

    def start_encoding_process(self, df, path):
        df.drop(columns=COLUMNS_TO_DROP_BEFORE_ENCODING)
        data = self.encode_session(df)
        if SAVE_ENCODED_PICTURES:
            self.save_encoded_picture(data, path)

    @staticmethod
    def save_encoded_picture(data, path):
        data.save(path)

    def encode_session(self, df):
        used_logs = df.head(self.number_of_packets_in_picture)
        missing = (self. number_of_packets_in_picture - used_logs.shape[0]) * len(df.columns)
        if missing:
            padded = [0] * missing

        val = list(used_logs.values)
        result = [v for sub in val for v in sub]
        used_logs = result + padded
        used_logs = [float(val) for val in used_logs]
        pixels = np.array(used_logs)
        pixels = np.reshape(pixels, (20, self.length_of_packet))
        pixels = pixels * 255
        pixels = pixels.astype(np.uint8)
        data = im.fromarray(pixels)
        return data
