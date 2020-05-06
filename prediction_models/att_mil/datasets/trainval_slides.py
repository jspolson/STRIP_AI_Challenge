import lmdb
import torch.utils.data as data
from PIL import Image


class BiopsySlides:
    def __init__(self, dataset_params, transform, data_list, phase='train'):
        self.data_list = data_list
        self.transform = transform
        self.params = dataset_params
        self.phase = phase
        self.tiles_env = lmdb.open(f"{dataset_params['tiles_dir']}", max_readers=3, readonly=True,
                                   lock=False, readahead=False, meminit=False)





