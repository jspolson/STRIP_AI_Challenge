import pandas as pd
import os
import shutil
import lmdb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from prediction_models.att_mil.utils import file_utils


# def convert_

# Generate tile label given a tile label mask
def generate_tile_label(label_mask_dir, lmdb_dir, mask_size, trainval_file, ):
    tile_label_data = []
    env_label_masks = lmdb.open(f"{lmdb_dir}/label_masks", max_readers=3, readonly=True, lock=False,
                                readahead=False, meminit=False)
    with env_label_masks.begin(write=False) as txn_labels:
        for tile_name, mask_buff in txn_labels.cursor():
            tile_name = str(tile_name.decode('ascii'))
            tile_mask = file_utils.decode_buffer(mask_buff, (mask_size, mask_size), np.uint8)

    return


# Generate n files for n fold cross validation
def generate_cv_split(trainval_file, out_dir, n_fold, seed, delete_dir=False):
    if not delete_dir and os.path.isdir(out_dir):
        print("Cross validation file already generate!")
        return

    if delete_dir or (not os.path.isdir(out_dir)):
        if delete_dir:
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    train_df = pd.read_csv(trainval_file)
    splits = StratifiedKFold(n_splits=n_fold, random_state=seed, shuffle=True)
    for fold, (train_ids, valid_ids) in enumerate(splits.split(train_df, train_df.isup_grade)):
        cur_train_df = train_df[train_ids]
        cur_val_df = train_df[valid_ids]
        cur_train_df.to_csv(f"{out_dir}/train_{fold}.csv")
        cur_val_df.to_csv(f"{out_dir}/val_{fold}.csv")
    print("Finish generate cross valiadtion file")
    return


