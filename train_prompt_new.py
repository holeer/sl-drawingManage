import torch
import time
from transformers import BertConfig, BertTokenizerFast, BertForMaskedLM
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from utils import read_dataset_prompt
from torch.utils.data import DataLoader
from config.config import config


# 定义模型
checkpoint = "model/bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
bert_config = BertConfig.from_pretrained(checkpoint)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class BERTModel(torch.nn.Module):
    def __init__(self, checkpoint, config):
        super(BERTModel, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(checkpoint, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logit = outputs[0]
        return logit


# 构建数据集
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, sentences, attention_mask, token_type_ids, label):
        super(MyDataSet, self).__init__()
        self.sentences = torch.LongTensor(sentences)
        self.attention_mask = torch.ByteTensor(attention_mask)
        self.token_type_ids = torch.LongTensor(token_type_ids)
        self.label = torch.LongTensor(label)

    def __len__(self):
        return self.sentences.shape[0]

    def __getitem__(self, idx):
        return self.sentences[idx], self.attention_mask[idx], self.token_type_ids[idx], self.label[idx]


# 构建dataset
def create_dataset(text, label, tokenizer, max_len):
    X_train, X_test, Y_train, Y_test = train_test_split(text, label, test_size=0.2, random_state=1)
    train_dict = {'text': X_train, 'label_text': Y_train}
    test_dict = {'text': X_test, 'label_text': Y_test}
    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)

    def preprocess_function(examples):
        input_ids = []
        token_type_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(examples)):
            encode_dict = tokenizer.encode_plus(examples[i]['text'], max_length=max_len, padding="max_length", truncation=True)
            input_ids.append(encode_dict["input_ids"])
            token_type_ids.append(encode_dict["token_type_ids"])
            attention_masks.append(encode_dict["attention_mask"])
            label_ids.append(tokenizer.encode_plus(examples[i]['label_text'], max_length=max_len, padding="max_length", truncation=True)['input_ids'])
        return input_ids, attention_masks, token_type_ids, label_ids

    input_ids, attention_masks, token_type_ids, label_ids = preprocess_function(train_dataset)
    train_dataset = MyDataSet(input_ids, attention_masks, token_type_ids, label_ids)
    input_ids, attention_masks, token_type_ids, label_ids = preprocess_function(test_dataset)
    test_dataset = MyDataSet(input_ids, attention_masks, token_type_ids, label_ids)
    # test_dataset = test_dataset.map(preprocess_function, batched=True)
    return train_dataset, test_dataset


# 训练函数
def train(net, train_iter, test_iter, lr, weight_decay, num_epochs):
    total_time = 0
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=len(train_iter), num_training_steps=num_epochs * len(train_iter)
    )
    for epoch in range(num_epochs):
        start_of_epoch = time.time()
        cor = 0
        loss_sum = 0
        net.train()
        for idx, (ids, att_mask, type, y) in enumerate(train_iter):
            optimizer.zero_grad()
            ids, att_mask, type, y = ids.to(device), att_mask.to(device), type.to(device), y.to(device)
            out_train = net(ids, att_mask, type)
            l = loss(out_train.view(-1, tokenizer.vocab_size), y.view(-1))
            l.backward()
            optimizer.step()
            schedule.step()
            loss_sum += l.item()
            if (idx + 1) % 1 == 0:
                print("Epoch {:04d} | Step {:06d}/{:06d} | Loss {:.4f} | Time {:.0f}".format(
                    epoch + 1, idx + 1, len(train_iter), loss_sum / (idx + 1), time.time() - start_of_epoch)
                )
            print(out_train.view(-1, tokenizer.vocab_size).size())
            print(att_mask)
            print(y)
            predicted = torch.max(out_train, att_mask)[1]
            cor += (predicted == y).sum()
            cor = float(cor)

        acc = float(cor / train_iter)

        eval_loss_sum = 0.0
        net.eval()
        correct_test = 0
        with torch.no_grad():
            for ids, att, tpe, y in test_iter:
                ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
                out_test = net(ids, att, tpe)
                loss_eval = loss(out_test.view(-1, tokenizer.vocab_size), y.view(-1))
                eval_loss_sum += loss_eval.item()
                predicted_test = torch.max(att_mask, 1)[1]
                correct_test += (predicted_test == y).sum()
                correct_test = float(correct_test)
        acc_test = float(correct_test / test_iter)

        if epoch % 1 == 0:
            print(("epoch {}, train_loss {},  train_acc {} , eval_loss {} ,acc_test {}".format(
                epoch + 1, loss_sum / (len(train_iter)), acc, eval_loss_sum / (len(test_iter)), acc_test))
            )
            train_loss.append(loss_sum / len(train_iter))
            eval_loss.append(eval_loss_sum / len(test_iter))
            train_acc.append(acc)
            eval_acc.append(acc_test)

        end_of_epoch = time.time()
        print("epoch {} duration:".format(epoch + 1), end_of_epoch - start_of_epoch)
        total_time += end_of_epoch - start_of_epoch

    print("total training time: ", total_time)


if __name__ == '__main__':
    # 构建数据集
    train_batch_size = 4
    test_batch_size = 4

    text, label = read_dataset_prompt(config.train_file, config.max_length)
    train_dataset, test_dataset = create_dataset(text, label, tokenizer, config.max_length)
    train_iter = DataLoader(train_dataset, train_batch_size, True)
    test_iter = DataLoader(test_dataset, test_batch_size, True)
    train_loss = []
    eval_loss = []
    train_acc = []
    eval_acc = []

    # 开始训练
    net = BERTModel(checkpoint, bert_config)
    num_epochs, lr, weight_decay = 10, 2e-5, 1e-4
    print("training...")
    train(net, train_iter, test_iter, lr, weight_decay, num_epochs)
