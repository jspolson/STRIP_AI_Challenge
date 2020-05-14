## system package
import os, sys
sys.path.append('../')
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

## general package
import torch
import torch.nn as nn
from fastai.vision import *
# from torch_lr_finder import LRFinder
from tqdm import trange, tqdm
from sklearn.metrics import cohen_kappa_score,confusion_matrix
## custom package
from input.inputPipeline import *
from model.resnext_ssl import *
class Train(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    def train_epoch(self,trainloader, valloader, criterion):
        ## train
        self.model.train()
        train_loss = []
        for i, data in enumerate(tqdm(trainloader, desc='trainIter'), start=0):
            #             if i >= 50:
            #                 break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
        ## val
        model.eval()
        val_loss, val_label, val_preds = [], [], []
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader, desc='valIter'), start=0):
                #                 if i > 50:
                #                     break
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs.cuda())
                loss = criterion(outputs, labels.cuda())
                val_loss.append(loss.item())
                val_label.append(labels.cpu())
                val_preds.append(outputs.cpu())
        # scheduler.step()
        val_preds = torch.argmax(torch.cat(val_preds, 0), 1)
        val_label = torch.cat(val_label)
        kappa = cohen_kappa_score(val_label, val_preds, weights='quadratic')
        return np.mean(train_loss), np.mean(val_loss), kappa

if __name__ == "__main__":
    nfolds = 5
    bs = 32
    epochs = 16
    csv_file = '../input/panda-16x128x128-tiles-data/{}_fold_train.csv'.format(nfolds)
    image_dir = '../input/panda-16x128x128-tiles-data/train/'
    ## image statistics
    # mean = torch.tensor([0.90949707, 0.8188697, 0.87795304])
    # std = torch.tensor([0.36357649, 0.49984502, 0.40477625])
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    ## image transformation
    tsfm = data_transform(mean, std)
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDataset(csv_file, image_dir, transform=tsfm)
    ## dataloader
    crossValData = crossValDataloader(csv_file, dataset, bs)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    Training = Train(model, optimizer)
    for fold in trange(nfolds, desc='fold'):
        model = Model().cuda()
        trainloader, valloader = crossValData(fold)
        for epoch in trange(epochs, desc='epoch'):
            train_loss, val_loss, kappa = Training.train_epoch(trainloader,valloader,criterion)
            # print("Epoch {}, train loss: {:.4f}, val loss: {:.4f}, kappa-score: {:.4f}.\n".format(epoch,
            #                                                                                    train_loss,
            #                                                                                    val_loss,
            #                                                                                    kappa))
            tqdm.write("Epoch {}, train loss: {:.4f}, val loss: {:.4f}, kappa-score: {:.4f}.\n".format(epoch,
                                                                                               train_loss,
                                                                                               val_loss,
                                                                                               kappa))