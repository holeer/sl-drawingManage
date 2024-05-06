# -*- coding: utf-8 -*-

import data_processing
import torch
import utils
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import model

vocabulary = utils.load_txt('dataset/vocabulary_label.txt')
tokenizer = BertTokenizer.from_pretrained("model/bert-base-chinese/")
bert_model = BertModel.from_pretrained("model/bert-base-chinese/")
res_model = models.resnet101(pretrained=True)

predict_model_path = 'snapshot/best_steps_30.pt'


def get_label(drawing, content):
    cnn = model.textCNN(len(vocabulary))
    cnn.load_state_dict(torch.load(predict_model_path))
    print('*' * 30 + '加载模型成功' + '*' * 30)
    feature = data_processing.feature_extract(res_model, bert_model, tokenizer, drawing, content)
    cnn.eval()
    logit = cnn(feature)
    logit[logit > 0.5] = 1
    logit[logit < 0.5] = 0
    label = utils.tensor2label(vocabulary, logit)
    return label


if __name__ == '__main__':
    # get_label('test.png', '改路路基横断面图I改路路基横断面图II注：1、本图尺寸单位以cm计，L为路基宽度。2、改路断面具体设计位置见改路工程数量表。3、改路路基横断面图I适用于原有挡墙道路，改路路基横断面图II适用于一般放坡道路。浙江省交通规划设计研究院G228宁海西店至桃源段公路工程第3合同改路横断面设计图设计傅广卷复核徐金胜审核吴锐图号201401090388S9-5-1')
    label = get_label('test.png', 'IV级围岩衬砌配筋图注：1.本图尺寸除钢筋直径以毫米计外，余均以厘米计。2.主筋净保护层厚为5cm。3.本图适用于IV级围岩衬砌（S4c型）结构配筋。浙江省交通规划设计研究院长春至深圳国家高速公路浙江省湖州段扩容工程第合同分离式隧道IV级围岩二次衬砌配筋图（七）设计吴根强复核丁海洋审核郑军华图号201701010050S5T-22-7')
    print(label)
