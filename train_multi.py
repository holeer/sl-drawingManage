# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import os
import tqdm
import model
import csv
import utils
import torch.utils.data as data
import dataset_multi
import argparse

#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#

parser = argparse.ArgumentParser(description='')
# learning
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=10, help='how many steps to wait before testing [default: 10]')
parser.add_argument('-save-interval', type=int, default=10, help='how many steps to wait before saving [default: 10]')
parser.add_argument('-save-dir', type=str, default='snapshot/', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
parser.add_argument('-drop-last', action='store_true', default=False, help='drop last batch of data')
parser.add_argument('-num-workers', type=int, default=0, help='number workers')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

vocabulary = utils.load_txt('dataset/vocabulary_label.txt')
num_classes = len(vocabulary)
total_test_size = 0


def train(model, args, train_iter, test_iter):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0
    print('*' * 30 + 'Training' + '*' * 30)
    for epoch in range(args.epochs):
        steps = 0
        model.train()
        optimizer.zero_grad()
        for batch in train_iter:
            feature, label = batch
            logit = model(feature)
            loss = nn.BCELoss()(logit, label)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                logit[logit > 0.5] = 1
                logit[logit < 0.5] = 0
                corrects = (logit.data == label.data).sum()
                accuracy = corrects / (len(batch[0]) * num_classes)
                print('\rEpoch[{}] Batch[{}] - loss: {:.6f} acc: {:.4f}'.format(epoch,
                                                                                steps,
                                                                                loss.data.item(),
                                                                                accuracy))

        test_acc = eval(model, args, test_iter)
        if test_acc > best_acc:
            best_acc = test_acc
            if args.save_best:
                save(model, args.save_dir, 'best', epoch)


def save(model, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)


def eval(model, args, data_iter):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, label = batch
        logit = model(feature)
        loss = nn.BCELoss()(logit, label)
        avg_loss += loss.item()
        logit[logit > 0.5] = 1
        logit[logit < 0.5] = 0
        corrects += (logit.data == label.data).sum()

    avg_loss /= total_test_size
    accuracy = corrects / (total_test_size * num_classes)
    print('\nEvaluation - loss: {:.6f} acc: {:.4f} \n'.format(avg_loss, accuracy))

    return accuracy


if __name__ == '__main__':
    model = model.textCNN(num_classes)
    train_path = 'dataset/train/train.csv'
    test_path = 'dataset/test/test.csv'

    train_data = dataset_multi.DrawingDataset(train_path)
    test_data = dataset_multi.DrawingDataset(test_path)
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        num_workers=args.num_workers
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        num_workers=args.num_workers
    )
    total_test_size = len(test_data)
    train(model, args, train_loader, test_loader)
