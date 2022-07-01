# -*- coding: UTF-8 -*-
import torch
from config.config import config
import utils
from get_entity import find_all_tag
from BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel
from PIL import Image


transform_list = [transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
img_to_tensor = transforms.Compose(transform_list)

bert_tokenizer = BertTokenizer.from_pretrained("model/bert-base-chinese/")
bert_model = BertModel.from_pretrained("model/bert-base-chinese/")
res_model = models.resnet101(pretrained=True)


class Entity_predict:
    """
    命名实体识别类
    """

    def __init__(self):
        self.vocab = utils.load_vocab(config.vocab)
        self.device = None
        self.model = None
        self.get_model()  # 模型准备

    def get_model(self):
        """
        构建模型
        :return:
        """
        label_dic = utils.load_vocab(config.label_file)
        tag_set_size = len(label_dic)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = BERT_BiLSTM_CRF(tag_set_size,
                                     config.bert_embedding,
                                     config.rnn_hidden,
                                     config.rnn_layer,
                                     config.dropout,
                                     config.pretrain_model_name,
                                     self.device).to(self.device)
        checkpoint = torch.load(config.checkpoint)
        self.model.load_state_dict(checkpoint["model"])

    def predict(self, pic_path, input_seq, position, max_length=config.max_length):
        """
        :return: 实体列表
        """
        # 构造输入
        input_list = []
        for i in range(len(input_seq)):
            input_list.append(input_seq[i])

        if len(input_list) > max_length - 2:
            input_list = input_list[-(max_length - 2):0]
            position = position[-(max_length - 2):0]
        input_list = ['[CLS]'] + input_list + ['[SEP]']
        input_ids = [int(self.vocab[word]) if word in self.vocab else int(self.vocab['[UNK]']) for word in input_list]
        input_mask = [1] * len(input_ids)
        position.insert(0, [0, 0, 0, 0])
        position.append([1, 1, 1, 1])

        if len(input_ids) < max_length:
            input_ids.extend([0] * (max_length - len(input_ids)))
            input_mask.extend([0] * (max_length - len(input_mask)))
            for i in range(max_length - len(position)):
                position.append([0, 0, 0, 0])
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(position) == max_length
        # 变为tensor并放到GPU上, 二维, 这里mask在CRF中必须为unit8类型或者bool类型
        input_ids = torch.LongTensor([input_ids]).to(self.device)
        input_mask = torch.ByteTensor([input_mask]).to(self.device)
        input_position = torch.LongTensor([position]).to(self.device)

        # 提取图像特征
        res_model.fc = torch.nn.Linear(2048, 768)
        res_model.eval()
        img = Image.open(pic_path)
        img = img.resize((224, 224))
        pic = img_to_tensor(img).resize_(1, 3, 224, 224)
        pic_output = res_model(Variable(pic))
        output_npy = pic_output.data.cpu().numpy()[0]
        pic_feature = []
        for i in range(max_length):
            pic_feature.append(output_npy)

        input_pic = torch.LongTensor([pic_feature]).to(self.device)

        feats = self.model(input_ids, input_mask, input_position, input_pic)
        # out_path是一条预测路径（数字列表）, [1:-1]表示去掉一头一尾, <START>和<EOS>标志
        model_out = self.model.predict(feats, input_mask)
        out_path = model_out[0][1:-1]
        res = find_all_tag(out_path)
        print(res)
        re = []
        for name in res:
            for i in res[name]:
                re.append(input_seq[i[0]:(i[0] + i[1])])
        return re


getEntity = Entity_predict()
res = getEntity.predict('test.png', "广B立面广c广90-61028M桥梁全长：36410第三联)430的0-12000预应力凝士（后张）侉支查连续T梁410430003000410160型仲笔缝0饰1115右侧地面线1601602B112)层理产：318°1491203z99中桩地面线112.中风化无质教岩715.7715.9平面改路4.5颜Z14+863政河B-22m天目溪桥梁中心格男建德人建德二接0版道改路1h4.5屋浙江省交通规划设计研究院有限公司临金高速公路临安至建德段工程於潜北互通(互通段)右线杭州市交通规划设计研究院第1标段设计曹林复核唐翔网审核朱突勤丸图号2017GL030373S6-5-4-2-3",
                        [])
print(res)
