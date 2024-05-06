# -*- coding: utf-8 -*-

import jieba
import os
import csv
import jieba.posseg
import jieba.analyse
from string import punctuation as punctuation_en
from zhon.hanzi import punctuation as punctuation_zh


def ocr_single(reader, img_path, canvas_size):
    img_list = os.listdir(img_path)
    img_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    with open("train.csv", "w", newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['drawing', 'content', 'label'])
        for i in img_list:
            result = reader.readtext(img_path + i, canvas_size=canvas_size)
            word_list = []
            for j in result:
                position = j[0]
                word = j[1]
                recognition_rate = j[2]
                for p in punctuation_en:
                    word = word.replace(p, '')
                for p in punctuation_zh:
                    word = word.replace(p, '')
                word = word.replace(' ', '')
                word_list.append(word)
            content = ''.join(word_list).replace(' ', '')
            if len(content) > 0:
                if int(i.split('_')[1].split('.')[0]) <= 194:
                    label = '公路工程/施工图设计/路线交叉'
                elif 194 < int(i.split('_')[1].split('.')[0]) <= 396:
                    label = '公路工程/施工图设计/路基路面'
                elif 396 < int(i.split('_')[1].split('.')[0]) <= 615:
                    label = '公路工程/施工图设计/桥梁涵洞'
                elif 615 < int(i.split('_')[1].split('.')[0]) <= 1048:
                    label = '公路工程/施工图设计/隧道'
                writer.writerow([img_path + i, content, label])
                print(i + "图纸识别完成...")


def text_analyze(word_str, method, top_number):
    if method == 0:
        with open("result_textrank.csv", "w", newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            print("*" * 30 + "Text Rank" + "*" * 30)
            for x, w in jieba.analyse.textrank(word_str, withWeight=True, topK=top_number):
                print('%s %s' % (x, w))
                writer.writerow([x, w])
    else:
        with open("result_tfidf.csv", "w", newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            print("*" * 30 + "TF-IDF" + "*" * 30)
            for x, w in jieba.analyse.extract_tags(word_str, withWeight=True, topK=top_number):
                print('%s %s' % (x, w))
                writer.writerow([x, w])
