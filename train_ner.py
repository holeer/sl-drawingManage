# -*- coding: utf-8 -*-

import os
import torch
from utils import load_vocab, read_dataset
from config.config import config
from config.logger import Logger
from torch.utils.data import DataLoader, TensorDataset
from BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
import time
import torch.nn as nn
from tqdm import tqdm
from estimate import Precision, Recall, F1_score


LOG_FILENAME = config.log_dir + "NER_" + str(int(time.time())) + ".log"
print(30 * "=",
      "Training log in file: {}".format(LOG_FILENAME),
      30 * "=")
log = Logger(filename=LOG_FILENAME)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)
    tag_set_size = len(label_dic)
    print('*' * 30 + '加载训练集' + '*' * 30)
    train_data = read_dataset(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    print('*' * 30 + '加载验证集' + '*' * 30)
    dev_data = read_dataset(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)

    train_ids = torch.LongTensor([temp.input_id for temp in train_data]).to(device)
    train_masks = torch.ByteTensor([temp.input_mask for temp in train_data]).to(device)
    train_tags = torch.LongTensor([temp.label_id for temp in train_data]).to(device)
    train_positions = torch.FloatTensor([temp.position for temp in train_data]).to(device)
    train_pics = torch.FloatTensor([temp.pic_feature for temp in train_data]).to(device)

    train_dataset = TensorDataset(train_ids, train_masks, train_tags, train_positions, train_pics)
    train_loader = DataLoader(train_dataset, shuffle=config.shuffle, batch_size=config.batch_size)

    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data]).to(device)
    dev_masks = torch.ByteTensor([temp.input_mask for temp in dev_data]).to(device)
    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data]).to(device)
    dev_positions = torch.FloatTensor([temp.position for temp in dev_data]).to(device)
    dev_pics = torch.FloatTensor([temp.pic_feature for temp in dev_data]).to(device)

    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags, dev_positions, dev_pics)
    dev_loader = DataLoader(dev_dataset, shuffle=config.shuffle, batch_size=config.batch_size)

    model = BERT_BiLSTM_CRF(tag_set_size,
                            config.bert_embedding,
                            config.rnn_hidden,
                            config.rnn_layer,
                            config.dropout,
                            config.pretrain_model_name,
                            device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=1)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    # _, valid_loss, start_estimator = valid(model,
    #                                        dev_loader)
    # print("\t* Validation loss before training: loss = {:.4f}, precision: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%"
    #       .format(valid_loss, (start_estimator[0] * 100), (start_estimator[1] * 100), (start_estimator[2] * 100)))

    # -------------------- Training epochs ------------------- #
    print(30 * "=",
          "Training BERT_BiLSTM_CRF model on device: {}".format(device),
          30 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, config.epochs + 1):
        epochs_count.append(epoch)

        log.logger.info("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss = train(model,
                                       train_loader,
                                       optimizer,
                                       config.max_grad_norm)
        train_losses.append(epoch_loss)
        log.logger.info("-> Training time: {:.4f}s, loss = {:.4f}"
              .format(epoch_time, epoch_loss))

        epoch_time, valid_loss, valid_estimator = valid(model,
                                                        dev_loader)
        valid_losses.append(valid_loss)
        log.logger.info("-> Valid time: {:.4f}s, loss = {:.4f}, precision: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%"
              .format(epoch_time, valid_loss, (valid_estimator[0] * 100), (valid_estimator[1] * 100),
                      (valid_estimator[2] * 100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(valid_estimator[2])

        # Early stopping on validation accuracy.  estimator[2]: F1
        if valid_estimator[2] < best_score:
            patience_counter += 1
        else:
            best_score = valid_estimator[2]
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(config.target_dir, "RoBERTa_best.pth.tar"))
            log.logger.info("saved the best model in epoch {}.".format(epoch))

        # if epoch % 1 == 0:
        #     # Save the model at each epoch.
        #     torch.save({"epoch": epoch,
        #                 "model": model.state_dict(),
        #                 "best_score": best_score,
        #                 "optimizer": optimizer.state_dict(),
        #                 "epochs_count": epochs_count,
        #                 "train_losses": train_losses,
        #                 "valid_losses": valid_losses},
        #                os.path.join(config.target_dir, "RoBERTa_NER_{}.pth.tar".format(epoch)))

        if patience_counter >= config.patience:
            log.logger.info("-> Early stopping: patience limit reached, stopping...")
            break


def train(model, dataloader, optimizer, max_gradient_norm):
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        inputs, masks, tags, positions, pics = batch

        inputs = inputs.to(device)
        masks = masks.byte().to(device)
        tags = tags.to(device)

        optimizer.zero_grad()
        feats = model(inputs, masks, positions, pics)
        loss = model.loss(feats, tags, masks)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    return epoch_time, epoch_loss


def valid(model, dataloader):
    model.eval()
    device = model.device
    pre_output = []
    true_output = []
    epoch_start = time.time()
    running_loss = 0.0

    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader)
        for _, batch in enumerate(tqdm_batch_iterator):
            inputs, masks, tags, positions, pics = batch

            real_length = torch.sum(masks, dim=1)
            tmp = []
            i = 0
            for line in tags.cpu().numpy().tolist():
                tmp.append(line[0: real_length[i]])
                i += 1

            true_output.append(tmp)

            inputs = inputs.to(device)
            masks = masks.byte().to(device)
            tags = tags.to(device)

            feats = model(inputs, masks, positions, pics)
            loss = model.loss(feats, tags, masks)
            out_path = model.predict(feats, masks)
            pre_output.append(out_path)

            running_loss += loss.item()
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    # 计算精确度、召回率、F1值
    precision = Precision(pre_output, true_output)
    recall = Recall(pre_output, true_output)
    f1_score = F1_score(precision, recall)

    estimator = (precision, recall, f1_score)

    return epoch_time, epoch_loss, estimator


if __name__ == '__main__':
    main()
