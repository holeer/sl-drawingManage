# -*- coding: utf-8 -*-

import torch
import torch.nn
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
from string import punctuation as punctuation_en
from zhon.hanzi import punctuation as punctuation_zh
import re


transform_list = [transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
img_to_tensor = transforms.Compose(transform_list)


def feature_extract(res_model, bert_model, tokenizer, drawing, content):
    # 提取图像特征
    res_model.fc = torch.nn.LeakyReLU(0.2)
    res_model.eval()
    img = Image.open(drawing)
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)
    tensor = tensor.resize_(1, 3, 224, 224)
    # tensor = tensor.cuda()
    result = res_model(Variable(tensor))
    result_npy = result.data.cpu().numpy()
    result_tensor = torch.from_numpy(result_npy[0])

    # 提取文字特征
    # BERT 文本长度不能超过512，加上标识符，文本长度在510内
    # for p in punctuation_en:
    #     content = content.replace(p, '')
    # for p in punctuation_zh:
    #     content = content.replace(p, '')
    # if len(content) > 510:
    #     content = re.sub('[\d]', '', content)
    # if len(content) > 510:
    #     content = re.sub('[a-zA-Z]', '', content)
    if len(content) > 510:
        content = content[-510:]
    inputs = tokenizer(content, return_tensors="pt")
    outputs = bert_model(**inputs)

    # 合并
    out = torch.cat([outputs[1].squeeze(), result_tensor], dim=0)
    return out


