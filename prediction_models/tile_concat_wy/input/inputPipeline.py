import os
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class crossValInx(object):
    def __init__(self, csv_file):
        self.crossVal_csv = pd.read_csv(csv_file)

    def __call__(self, fold = 0):
        val_idx = self.crossVal_csv.index[self.crossVal_csv['split'] == fold].tolist()
        train_idx = list(set([x for x in range(len(self.crossVal_csv))]) - set(val_idx))
        return train_idx, val_idx

class crossValDataloader(object):
    def __init__(self, csv_file, dataset, bs = 4):
        self.inx = crossValInx(csv_file)
        self.dataset = dataset
        self.bs = bs

    def __call__(self, fold = 0):
        train_idx, val_idx = self.inx(fold)
        train = torch.utils.data.Subset(self.dataset, train_idx)
        val = torch.utils.data.Subset(self.dataset, val_idx)
        trainloader = torch.utils.data.DataLoader(train, batch_size=self.bs, shuffle=True, num_workers=4,
                                                  collate_fn=dataloader_collte_fn, pin_memory=True)
        valloader = torch.utils.data.DataLoader(val, batch_size=self.bs, shuffle=True, num_workers=4,
                                                collate_fn=dataloader_collte_fn, pin_memory=True)
        return trainloader, valloader

class PandaPatchDataset(Dataset):
    """Panda Tile dataset. With fixed tiles for each slide."""
    def __init__(self, csv_file, image_dir, N = 12, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            N (interger): Number of tiles selected for each slide.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_csv = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.N = N

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fnames = [os.path.join(self.image_dir, self.train_csv.loc[idx, 'image_id']+'_'+str(i)+'.png')
                  for i in range(self.N)]
        imgs = [self.open_image(fname)
               for fname in fnames]
        isup_grade = self.train_csv.loc[idx, 'isup_grade']

        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        ## convert the output to tensor
        imgs = [torch.tensor(img) for img in imgs]
        imgs = torch.stack(imgs)
        isup_grade = torch.tensor(isup_grade)
        sample = {'image': imgs, 'isup_grade': isup_grade}
        return sample

    def open_image(self, fn, convert_mode='RGB', after_open=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
            x = Image.open(fn).convert(convert_mode)
        if after_open:
            x = after_open(x)
        return x

def data_transform(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)):
    tsfm = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomAffine(degrees=180, fillcolor=(255, 255, 255)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean,
                              std=std)])
    return tsfm

def dataloader_collte_fn(batch):
    imgs = [item['image'] for item in batch]
    imgs = torch.stack(imgs)
    target = [item['isup_grade'] for item in batch]
    target = torch.stack(target)
    return [imgs, target]

if __name__ == "__main__":
    ## input files and folders
    nfolds = 5
    bs = 4
    csv_file = './panda-16x128x128-tiles-data/{}_fold_train.csv'.format(nfolds)
    image_dir = './panda-16x128x128-tiles-data/train/'
    ## image statistics
    mean = torch.tensor([0.90949707, 0.8188697, 0.87795304])
    std = torch.tensor([0.36357649, 0.49984502, 0.40477625])
    ## image transformation
    tsfm = data_transform(mean, std)
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDataset(csv_file, image_dir, transform=tsfm)
    ## dataloader
    dataloader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=4, collate_fn=dataloader_collte_fn)

    ## fetch data from dataloader
    img, target = iter(dataloader).next()
    print("image size:{}, target sise:{}.".format(img.size(), target.size()))

    ## cross val dataloader
    crossValData = crossValDataloader(csv_file, dataset, bs)
    trainloader, valloader = crossValData(0)
    img, target = iter(trainloader).next()
    print("image size:{}, target sise:{}.".format(img.size(), target.size()))