# -*- coding: utf-8 -*-
import csv

from api_call import baidu_api
import jieba
import jieba.posseg as pseg
import utils
import difflib
import re
from PyPDF2 import PdfFileReader, PdfFileWriter
import fitz
from tqdm import tqdm
import os
import predict
from config.config import config


jieba.load_userdict(config.dict_path)
person_list = utils.get_person(config.dict_path)


def correct_name(word):
    for person in person_list:
        if difflib.SequenceMatcher(None, word, person).quick_ratio() >= 0.6:
            return person
    return word


def extract_img(token, img_path):
    drawing_dict = {}
    person_name_flag = 0
    drawing_name_flag = 0
    drawing_num_flag = 0
    project_name_flag = 0
    name_list = []
    sketch_list = []
    sheet_list = []
    drawing_num_list = []
    annotation_list = []
    texts, locations = baidu_api.ocr(token, img_path)
    if texts is not None:
        for i in reversed(texts):
            if drawing_num_flag != 1:
                drawing_num_list.append(i)
                if '图号' in i:
                    drawing_num_flag = 1
                    drawing_num = ' '.join(drawing_num_list)
                    drawing_dict['drawing_num'] = drawing_num.replace('图号', '')
            if '图号' in i:
                person_name_flag = 1
            if '设计' in i and person_name_flag == 1:
                name_list.append(i)
                person_name_flag = 0
            if person_name_flag == 1:
                name_list.append(i)
            if '工程' in i and project_name_flag != 1:
                project_name_flag = 1
                i = i.split('工程')[0] + '工程'
                if '研究院' in i:
                    drawing_dict['project_name'] = i.split('研究院')[1]
            if '图' in i and '图号' not in i and drawing_name_flag != 1:
                drawing_name_flag = 1
                if ')' in i:
                    drawing_dict['drawing_name'] = i.split(')')[0] + ')'
                else:
                    drawing_dict['drawing_name'] = i.split('图')[0] + '图'
            if ('示意' in i or '大样' in i or '设计图' in i or '截面' in i or '剖面' in i or '平面' in i or '立面' in i or '配置图' in i or '布置图' in i or '断面' in i) and len(i) <= 18:
                sketch_list.append(i)
            if '数量表' in i or '尺寸表' in i or '材料表' in i:
                sheet_list.append(i)

            rule = r'^[1-9][.][\u4e00-\u9fa5]'
            match = re.match(rule, i)
            if match:
                annotation_list.append(i)
        annotation_list.sort(key=lambda x: (int((re.search(r"([1-9]+)", x)).group(0))))
        drawing_dict['annotation'] = ''.join(annotation_list)
        drawing_dict['sketch'] = '/'.join(sketch_list)
        drawing_dict['sheet'] = '/'.join(sheet_list)

        names = ''.join(reversed(name_list))
        words = pseg.cut(names, use_paddle=True)
        designer_index, viewer_index, reviewer_index = 0, 0, 0
        for index, (word, flag) in enumerate(words):
            word = word.strip()
            if '设计' in word:
                designer_index = index + 1
            if '复核' in word:
                reviewer_index = index + 1
            if '审核' in word:
                viewer_index = index + 1

            if index == designer_index:
                drawing_dict['designer'] = correct_name(word)

            if index == viewer_index:
                drawing_dict['viewer'] = correct_name(word)
            if index == reviewer_index:
                drawing_dict['reviewer'] = correct_name(word)

        return drawing_dict, texts, locations
    else:
        return drawing_dict, [], []


def extract(file_path):
    token = baidu_api.fetch_token()
    result_list = []

    file_name = file_path.split(config.temp_path)[1]

    utils.init_dirs(config.pdf_path)
    utils.init_dirs(config.img_path)
    utils.init_dirs(config.txt_path)

    if file_path.split('.')[1] == 'png':
        print('*' * 30 + '开始OCR识别' + '*' * 30)
        utils.resize_pic(file_path, config.base_size)
        ocr_dict, texts, locations = extract_img(token, file_path)
        print('*' * 30 + '开始预测图纸标签' + '*' * 30)
        content = ''.join(texts).replace(' ', '')
        label = predict.get_label(file_path, content)
        ocr_dict['label'] = label
        ocr_dict['file_name'] = file_name
        result_list.append(ocr_dict)
    elif file_path.split('.')[1] == 'pdf':
        print("*" * 30 + "开始拆分PDF" + "*" * 30)
        file_reader = PdfFileReader(file_path)
        total_pages = file_reader.getNumPages()
        for page in tqdm(range(total_pages)):
            # 实例化对象
            file_writer = PdfFileWriter()
            # 将遍历的每一页添加到实例化对象中
            file_writer.addPage(file_reader.getPage(page))
            with open("pdf/{}.pdf".format(page), 'wb') as out:
                file_writer.write(out)

        pdf_list = os.listdir(config.pdf_path)
        pdf_list.sort(key=lambda x: int(x.split('.')[0]))
        print("*" * 30 + "开始转换图片" + "*" * 30)
        for p in tqdm(pdf_list):
            pdf = fitz.open(config.pdf_path + p)
            page = pdf[0]
            trans = fitz.Matrix(4, 4).prerotate(0)
            pm = page.get_pixmap(matrix=trans, alpha=False)
            # 开始写图像
            pm.save(config.img_path + p.split(".")[0] + ".png")
            utils.resize_pic(config.img_path + p.split(".")[0] + ".png", config.base_size)
            pdf.close()

        img_list = os.listdir(config.img_path)
        img_list.sort(key=lambda x: int(x.split('.')[0]))
        print('*' * 30 + '开始OCR识别' + '*' * 30)
        for i in tqdm(img_list):
            ocr_dict, texts, locations = extract_img(token, config.img_path + i)

            content = ''.join(texts).replace(' ', '')
            label = predict.get_label(config.img_path + i, content) if len(content) > 0 else []
            ocr_dict['label'] = label
            ocr_dict['file_name'] = file_name
            result_list.append(ocr_dict)

        # utils.save_json_file(result_list, 'result.json')
    return result_list
