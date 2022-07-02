from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import utils
from transformers import BertTokenizer, BertModel
import data_processing
import torchvision.models as models

vocabulary = utils.load_txt('dataset/vocabulary_label.txt')
tokenizer = BertTokenizer.from_pretrained("model/bert-base-chinese/")
bert_model = BertModel.from_pretrained("model/bert-base-chinese/")
res_model = models.resnet101(pretrained=True)


class DrawingDataset(Dataset):

    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.labels = np.asarray(self.data['label'])
        self.drawings = np.asarray(self.data['drawing'])
        self.contents = np.asarray(self.data['content'])

    def __getitem__(self, item):
        feature = data_processing.feature_extract(res_model, bert_model, tokenizer, self.drawings[item], self.contents[item])
        label = self.labels[item]
        onehot = utils.onehot_label(vocabulary, label)
        target = torch.Tensor(onehot)
        return feature, target

    def __len__(self):
        return len(self.data.index)
