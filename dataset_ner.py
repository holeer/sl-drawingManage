# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import utils
from transformers import BertTokenizer, BertModel
import data_processing
import torchvision.models as models
from utils import read_dataset, load_vocab
from config.config import config


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese/")
bert_model = BertModel.from_pretrained(config.pretrain_model_name)
res_model = models.resnet101(pretrained=True)
label_dic = load_vocab(config.label_file)
vocab = load_vocab(config.vocab)


class NERDataset(Dataset):

    def __init__(self, dataset_path):
        self.data = read_dataset(path=dataset_path, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
        self.label_ids = np.array(self.data['label_id'])
        self.input_ids = np.array(self.data['input_id'])
        self.masks = np.array(self.data['mask'])
        self.pics = np.array(self.data['pic'])
        self.positions = np.array(self.data['position'])

    def __getitem__(self, item):
        features = data_processing.merge_features(res_model, bert_model, self.input_ids[item], self.masks[item], self.label_ids[item], self.pics[item], self.positions[item])
        tags = torch.Tensor(self.label_ids[item])
        masks = torch.Tensor(self.masks[item])
        return features, tags, masks

    def __len__(self):
        return len(self.pics)


if __name__ == '__main__':
    train_data = NERDataset(config.train_file)
    import torch.utils.data as data
    train_loader = data.DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    for batch in train_loader:
        print(batch)
        features, tags, masks = batch
