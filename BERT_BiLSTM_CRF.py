#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/3 16:54
# @Author  : JJkinging
# @File    : BERT_BiLSTM_CRF.py

import torch.nn as nn
import torch
from transformers import BertModel
from torchcrf import CRF
import torchvision.models as models
from config.config import config


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_set_size, embedding_dim, hidden_dim, rnn_layers, dropout, pretrain_model_name, device):
        """
        the model of BERT_BiLSTM_CRF
        :param bert_config:
        :param tag_set_size:
        :param embedding_dim:
        :param hidden_dim:
        :param rnn_layers:
        :param lstm_dropout:
        :param dropout:
        :param use_cuda:
        :return:
        """
        super(BERT_BiLSTM_CRF, self).__init__()
        self.tag_set_size = tag_set_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.device = device
        self.word_embeds = BertModel.from_pretrained(pretrain_model_name)
        for param in self.word_embeds.parameters():
            param.requires_grad = True
        self.LSTM = nn.LSTM(self.embedding_dim,
                            self.hidden_dim,
                            num_layers=self.rnn_layers,
                            bidirectional=True,
                            batch_first=True)
        self._dropout = nn.Dropout(p=self.dropout)
        self.CRF = CRF(num_tags=self.tag_set_size, batch_first=True)
        self.Liner = nn.Linear(self.hidden_dim*2, self.tag_set_size)
        self.Liner2 = nn.Linear(768 * 2 + 4, 768)

    def _init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return (torch.randn(2*self.rnn_layers, batch_size, self.hidden_dim).to(self.device),
                torch.randn(2*self.rnn_layers, batch_size, self.hidden_dim).to(self.device))

    def forward(self, sentence, attention_mask=None, positions=None, pics=None):
        '''
        :param sentence: sentence (batch_size, max_seq_len) : word-level representation of sentence
        :param attention_mask:
        :param positions:
        :param pics:
        :return: List of list containing the best tag sequence for each batch.
        '''
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        # embeds: [batch_size, max_seq_length, embedding_dim]
        embeds = self.word_embeds(sentence, attention_mask=attention_mask)[0]
        mix = torch.cat((embeds, pics), 2)
        mix = torch.cat((mix, positions), 2)
        text_pic_feature = self.Liner2(mix)
        # print(text_pic_feature.size())
        hidden = self._init_hidden(batch_size)

        # lstm_out: [batch_size, max_seq_length, hidden_dim*2]
        lstm_out, hidden = self.LSTM(text_pic_feature, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
        d_lstm_out = self._dropout(lstm_out)
        l_out = self.Liner(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

    def loss(self, feats, tags, mask):
        loss_value = self.CRF(emissions=feats,
                              tags=tags,
                              mask=mask,
                              reduction='mean')
        return -loss_value

    def predict(self, feats, attention_mask):
        # 做验证和测试时用
        out_path = self.CRF.decode(emissions=feats, mask=attention_mask)
        return out_path


