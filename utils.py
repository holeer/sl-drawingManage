# -*- coding: utf-8 -*-
import csv
import json
import math
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import jieba
import random
import torch.nn.functional as F
from collections import Counter
import os
from config.config import config
import shutil
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform_list = [transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
img_to_tensor = transforms.Compose(transform_list)

bert_tokenizer = BertTokenizer.from_pretrained("model/bert-base-chinese/")
bert_model = BertModel.from_pretrained("model/bert-base-chinese/")
res_model = models.resnet101(pretrained=True).to(device)
res_model.fc = torch.nn.Linear(2048, config.bert_embedding).to(device)
res_model.eval()


class InputFeature(object):
    def __init__(self, input_id, label_id, input_mask, position, pic_feature):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask
        self.position = position
        self.pic_feature = pic_feature


class MixedFeature(object):
    def __init__(self, label_id, input_mask, mixed_feature):
        self.label_id = label_id
        self.input_mask = input_mask
        self.mixed_feature = mixed_feature


def csv2json(csv_path, json_path, field_names):
    result_dict = {}
    result_list = []
    csv_file = open(csv_path, 'r', encoding='utf-8')
    json_file = open(json_path, 'w', encoding='utf-8')
    reader = csv.DictReader(csv_file, field_names)
    for row in reader:
        if reader.line_num == 1:
            continue
        label_list = row['label'].split('/')
        row['label'] = label_list
        result_list.append(row)
    result_dict['result'] = result_list
    out = json.dumps(result_dict, ensure_ascii=False)
    # out = json.dumps([row for row in reader], ensure_ascii=False)
    json_file.write(out)


def select(data, ids):
    return [data[i] for i in ids]


def load_txt(file):
    with open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        print("Load label data from file (%s) finished !" % file)
    return lines


def load_vocabulary(file_vocabulary_label):
    """
    Load vocabulary to dict
    """
    vocabulary = load_txt(file_vocabulary_label)
    dict_id2label, dict_label2id = {}, {}
    for i, l in enumerate(vocabulary):
        dict_id2label[str(i)] = str(l)
        dict_label2id[str(l)] = str(i)
    return dict_id2label, dict_label2id


def onehot_label(vocabulary, label):
    label_onehot = [0 for _ in range(len(vocabulary))]
    for i in label.split('/'):
        index = vocabulary.index(i)
        label_onehot[index] = 1
    return label_onehot


def tensor2label(vocabulary, logit):
    label_list = []
    output = list(map(int, logit.tolist()))
    for index, label in enumerate(vocabulary):
        if output[index] == 1:
            label_list.append(label)
    return '/'.join(label_list)


def shuffle_two(a1, a2):
    """
    Shuffle two list
    """
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    a1_ = [a1[l] for l in ran]
    a2_ = [a2[l] for l in ran]
    return a1_, a2_


def get_person(dict_path):
    person_list = []
    with open(dict_path, "r", encoding='utf-8') as f:  # 打开文件
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            if 'PER' in line:
                person_list.append(line.split(' ')[0])

    return person_list


def init_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def resize_pic(path, base_size):
    temp = Image.open(path)
    img_width = temp.width
    img_height = temp.height
    if img_width > base_size:
        scale = (base_size / img_width)
        h_size = int(img_height * scale)
        img = temp.resize((base_size, h_size), Image.ANTIALIAS)
        img.save(path)
        return base_size, h_size
    elif img_height > base_size:
        scale = (base_size / img_height)
        w_size = int(img_width * scale)
        img = temp.resize((w_size, base_size), Image.ANTIALIAS)
        img.save(path)
        return w_size, base_size
    else:
        return img_width, img_height


# 获得相对位置（四个点坐标）
def get_location(locations, width, height):
    location = []
    dw = 1. / width
    dh = 1. / height
    x = (2 * locations['left'] + locations['width']) / 2.0
    y = (2 * locations['top'] + locations['height']) / 2.0
    x = x * dw
    w = locations['width'] * dw
    y = y * dh
    h = locations['height'] * dh
    location.append(str(x))
    location.append(str(y))
    location.append(str(w))
    location.append(str(h))
    return ','.join(location)


def clear_space(input_path, output_path):
    last_index = 0
    txt_file = open(input_path, 'r', encoding='utf-8')
    new_file = open(output_path, 'w', encoding='utf-8')
    for index, value in enumerate(txt_file):
        # value = value.strip('\n')
        if value.strip('\n') == '':
            if last_index + 1 == index:
                continue
            else:
                last_index = index

        new_file.write(value)

    txt_file.close()
    new_file.close()


def tokenizer(s, word=False):
    if word:
        r = [w for w in s]
    else:
        s = jieba.cut(s, cut_all=False)
        r = " ".join(s).split()
    return r


def build_vocab(input_path, output_path, word):
    specials = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    counter = Counter()
    out_file = open(output_path, 'w', encoding='utf-8')
    with open(input_path, encoding='utf-8') as f:
        for string_ in f:
            counter.update(tokenizer(string_.strip(), word))

    words = list(counter)
    vocab_list = specials + words
    for i in vocab_list:
        out_file.writelines(i)
        out_file.write('\n')
    return vocab_list


def full_word_txt(txt_path, out_path):
    txt_list = os.listdir(txt_path)
    txt_list.sort(key=lambda x: int(x.split('.')[0]))
    word_list = []
    new_file = open(out_path, 'w', encoding='utf-8')

    for i in tqdm(txt_list):
        txt_file = open(txt_path + i, 'r', encoding='utf-8')

        for line in txt_file:
            line = line.strip()
            word = line.split(' ')
            word_list.append(word[0])

        content = ''.join(word_list).replace(' ', '')
        if len(content) == 0:
            content = '空白图纸'
        new_file.write(content)
        new_file.write('\n')
        new_file.write('\n')
        word_list.clear()


# def add_position_info(input_path, output_path):
#     in_file = open(input_path, 'r', encoding='utf-8')
#     file_index = 0
#     word_list = []
#     label_list = []
#     line_index = 0
#     for index, value in enumerate(in_file):
#         value = value.strip()
#         if len(value) == 0:
#             if file_index == 828 or file_index == 830 or file_index == 832:
#                 file_index += 1
#                 continue
#             txt_file = open('dataset/train/txt/' + str(file_index) + '.txt', 'r', encoding='utf-8')
#             out_file = open(output_path + str(file_index) + '.txt', 'w', encoding='utf-8')
#             for i, line in enumerate(txt_file):
#                 line = line.strip()
#                 word = line.split(' ')[0]
#                 position = line.split(' ')[1]
#                 if i < len(word_list) and word == word_list[line_index]:
#                     out_file.write(word + ' ' + label_list[line_index] + ' ' + position)
#                     out_file.write('\n')
#                     line_index += 1
#                 else:
#                     continue
#             print(str(file_index) + '.txt saving done.....')
#             file_index += 1
#             line_index = 0
#             word_list.clear()
#             label_list.clear()
#         else:
#             word_list.append(value.split(' ')[0])
#             label_list.append(value.split(' ')[1])


def add_position_info(label_file, position_file, output_path):
    in_file = open(label_file, 'r', encoding='utf-8')
    word_list = []
    label_list = []
    file_index = 0
    for index, value in enumerate(in_file):
        value = value.strip()
        if len(value) == 0:
            txt_file = open(position_file + str(file_index) + '.txt', 'r', encoding='utf-8')
            out_file = open(output_path + str(file_index) + '.txt', 'w', encoding='utf-8')
            for i, line in enumerate(txt_file):
                t = line.strip().split(' ')
                word = t[0]
                position = t[1]
                if word == word_list[i]:
                    out_file.write(word + ' ' + label_list[i] + ' ' + position)
                    out_file.write('\n')
            print(str(file_index) + '.txt saving done.....')
            file_index += 1
            word_list.clear()
            label_list.clear()
        else:
            content = value.split(' ')
            word_list.append(content[0])
            label_list.append(content[1])


def load_vocab(vocab_file):
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[token] = index
            index += 1
    return vocab


def get_area_feature(img, positions):
    pic_feature = []
    width, height = img.size
    zero_npy = np.zeros((config.bert_embedding, ), dtype=float)
    last_box = ()
    last_npy = np.zeros((config.bert_embedding, ), dtype=float)
    for index, i in enumerate(positions):
        if [0, 0, 0, 0] == i or [1, 1, 1, 1] == i:
            # temp = img.resize((224, 224))
            # pic = img_to_tensor(temp).resize_(1, 3, 224, 224)
            # pic_output = res_model(Variable(pic))
            # output_npy = pic_output.data.cpu().numpy()[0]
            # pic_feature.append(output_npy)
            pic_feature.append(zero_npy)
        elif [-1, -1, -1, -1] == i:
            pic_feature.append(zero_npy)
        else:
            area_width = width * i[2]
            area_height = height * i[3]
            x_min = (width * i[0]) - (area_width / 2)
            y_min = (height * i[1]) - (area_height / 2)
            x_max = x_min + area_width
            y_max = y_min + area_height
            box = (x_min, y_min, x_max, y_max)
            if box != last_box:
                last_box = box
                region = img.crop(box)
                region = region.resize((224, 224))
                pic = img_to_tensor(region).resize_(1, 3, 224, 224).to(device)
                pic_output = res_model(Variable(pic).to(device)).to(device)
                output_npy = pic_output.data.cpu().numpy()[0]
                last_npy = output_npy
                pic_feature.append(output_npy)
            else:
                pic_feature.append(last_npy)
    return pic_feature


def read_dataset(path, max_length, label_dic, vocab):
    data_list = os.listdir(path + 'txt/')
    data_list.sort(key=lambda x: int(x.split('.')[0]))
    result = []
    for data in tqdm(data_list):
        with open(path + 'txt/' + data, 'r', encoding='utf-8') as fp:
            words = []
            labels = []
            positions = []
            for line in fp:
                contents = line.strip()
                tokens = contents.split(' ')
                words.append(tokens[0])
                labels.append(tokens[1])
                position_list = []
                # if tokens[2] == 'N5':
                if '0.' != tokens[2][:2]:
                    error = [-1, -1, -1, -1]
                    position_list = error
                else:
                    position_list = [float(p) for p in tokens[2].split(',')]
                # print(position_list)
                positions.append(position_list)
            if len(words) > 0:
                if len(words) > max_length - 2:
                    words = words[-(max_length - 2):]
                    labels = labels[-(max_length - 2):]
                    positions = positions[-(max_length - 2):]
                words = ['[CLS]'] + words + ['[SEP]']
                labels = ['<START>'] + labels + ['<EOS>']
                positions.insert(0, [0, 0, 0, 0])
                positions.append([1, 1, 1, 1])
                input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in words]
                label_ids = [label_dic[i] for i in labels]
                input_mask = [1] * len(input_ids)
                # 填充
                if len(input_ids) < max_length:
                    input_ids.extend([0] * (max_length - len(input_ids)))
                    label_ids.extend([0] * (max_length - len(label_ids)))
                    input_mask.extend([0] * (max_length - len(input_mask)))
                    for i in range(max_length - len(positions)):
                        positions.append([0, 0, 0, 0])
                assert len(input_ids) == max_length
                assert len(label_ids) == max_length
                assert len(input_mask) == max_length
                assert len(positions) == max_length

                # 提取图像特征
                img = Image.open(path + 'img/' + str(data.split('.')[0]) + '.png')
                # pic_feature = get_area_feature(img, positions)
                img = img.resize((224, 224))
                pic = img_to_tensor(img).resize_(1, 3, 224, 224).to(device)
                pic_output = res_model(Variable(pic).to(device)).to(device)
                output_npy = pic_output.data.cpu().numpy()[0]
                pic_feature = []
                zero_npy = np.zeros((config.bert_embedding,), dtype=float)
                for i in input_ids:
                    # if i == 0 or i == int(vocab['[CLS]']) or i == int(vocab['[SEP]']):
                    if i == 0:
                        pic_feature.append(zero_npy)
                    else:
                        pic_feature.append(output_npy)

                features = InputFeature(input_id=input_ids, label_id=label_ids, input_mask=input_mask,
                                        position=positions, pic_feature=pic_feature)
                result.append(features)
    return result


def make_voc_dataset():
    import xml.dom.minidom as xmldom
    dict_id2label, dict_label2id = load_vocabulary('VOCdevkit/classes.txt')
    percents = [0.8]  # 指定训练集百分比0~1，其余是验证
    for percent in percents:
        img_files = os.listdir('VOCdevkit/JPEGImages/')
        img_files.sort(key=lambda x: int(x.split('.')[0]))
        split = int(len(img_files) * percent)
        train_img_files, val_img_files = img_files[:split], img_files[split:]
        with open('VOCdevkit/ImageSets/Main/train.txt', 'w', encoding='utf-8') as f1:
            for index, img_file in enumerate(train_img_files):
                root = xmldom.parse('VOCdevkit/Annotations/' + str(index) + '.xml').documentElement
                label_list = []
                for i in range(len(root.getElementsByTagName("name"))):
                    label = root.getElementsByTagName("name")[i].firstChild.data
                    id = dict_label2id[label]
                    xmin = root.getElementsByTagName("xmin")[i].firstChild.data
                    ymin = root.getElementsByTagName("ymin")[i].firstChild.data
                    xmax = root.getElementsByTagName("xmax")[i].firstChild.data
                    ymax = root.getElementsByTagName("ymax")[i].firstChild.data
                    temp = str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(id)
                    label_list.append(temp)
                img_name = 'VOCdevkit/JPEGImages/' + img_file
                f1.write(img_name + ' ' + ' '.join(label_list))
                f1.write('\n')
        print('train.txt done')
        with open('VOCdevkit/ImageSets/Main/val.txt', 'w', encoding='utf-8') as f2:
            for img_file in val_img_files:
                index = img_file.split('.')[0]
                root = xmldom.parse('VOCdevkit/Annotations/' + str(index) + '.xml').documentElement
                label_list = []
                for i in range(len(root.getElementsByTagName("name"))):
                    label = root.getElementsByTagName("name")[i].firstChild.data
                    id = dict_label2id[label]
                    xmin = root.getElementsByTagName("xmin")[i].firstChild.data
                    ymin = root.getElementsByTagName("ymin")[i].firstChild.data
                    xmax = root.getElementsByTagName("xmax")[i].firstChild.data
                    ymax = root.getElementsByTagName("ymax")[i].firstChild.data
                    temp = str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(id)
                    label_list.append(temp)
                img_name = 'VOCdevkit/JPEGImages/' + img_file
                f2.write(img_name + ' ' + ' '.join(label_list))
                f2.write('\n')
        print('val.txt done')


def read_log(path):
    for line in open(path, "r", encoding='utf-8'):
        print(line)


def read_dataset_prompt(path, max_length):
    data_list = os.listdir(path + 'txt/')
    data_list.sort(key=lambda x: int(x.split('.')[0]))
    print('*' * 30 + '加载数据集' + '*' * 30)
    texts = []
    labels = []
    for data in tqdm(data_list):
        with open(path + 'txt/' + data, 'r', encoding='utf-8') as fp:
            words = []
            drawing = []
            project = []
            design = []
            check = []
            recheck = []
            sheet = []
            num = []
            sketch = []
            for line in fp:
                contents = line.strip()
                tokens = contents.split(' ')
                words.append(tokens[0])
                if 'drawing' in tokens[1]:
                    drawing.append(tokens[0])
                if 'project' in tokens[1]:
                    project.append(tokens[0])
                if 'design' in tokens[1]:
                    design.append(tokens[0])
                if 'check' in tokens[1]:
                    check.append(tokens[0])
                if 'recheck' in tokens[1]:
                    recheck.append(tokens[0])
                if 'sheet' in tokens[1]:
                    sheet.append(tokens[0])
                if 'num' in tokens[1]:
                    num.append(tokens[0])
                if 'sketch' in tokens[1]:
                    sketch.append(tokens[0])

            content = ''.join(words)
            # prompt
            text = content + '。上文中，项目是' + '[MASK]'*len(project) + '，图名是' + '[MASK]'*len(drawing) + '，设计是' + '[MASK]'*len(design) + '，复核是' + '[MASK]'*len(recheck) + '，审核是' + '[MASK]'*len(check) + '，图号是' + '[MASK]'*len(num) + '，表格是' + '[MASK]'*len(sheet) + '，图示是' + '[MASK]'*len(sketch)
            label = content + '。上文中，项目是' + ''.join(project) + '，图名是' + ''.join(drawing) + '，设计是' + ''.join(design) + '，复核是' + ''.join(recheck) + '，审核是' + ''.join(check) + '，图号是' + ''.join(num) + '，表格是' + ''.join(sheet) + '，图示是' + ''.join(sketch)
            if len(content) > max_length:
                content = content[-(max_length - 2):]
                label = label[-(max_length - 2):]
            texts.append(text)
            labels.append(label)
    return texts, labels
