import lmdb
import numpy as np


def read_lmdb(lmdb_dir, data_shape, keys, data_type=np.uint8):
    return


def decode_buffer(buff, data_type, data_shape):
    buff = np.frombuffer(buff, dtype=data_type)
    buff = buff.reshape(data_shape)
    return buff
