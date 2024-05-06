# -*- coding: utf-8 -*-

import easyocr
import os
import cv2
import numpy as np
import difflib
from shapely import geometry
import jieba
import jieba.posseg as pseg
import pandas as pd
from PIL import Image
import json


# jieba.enable_paddle()


def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)


# 获取关键字开始位置
def keyword_start(word_list, position_list, keyword):
    for word in word_list:
        if difflib.SequenceMatcher(None, word, '备注').quick_ratio() >= 0.7 or difflib.SequenceMatcher(None, word, '注浆').quick_ratio() >= 0.6:
            continue
        similarity = difflib.SequenceMatcher(None, word, keyword).quick_ratio()
        # 与关键字相似度超过50%
        if similarity >= 0.5:
            # 从文本框左下角开始
            return position_list[word_list.index(word)]


def get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list):
    selected_list = []
    # 检测范围,逆时针写入坐标
    area = [(left_bottom_x, left_bottom_y), (right_top_x, left_bottom_y), (right_top_x, right_top_y),
            (left_bottom_x, right_top_y)]
    for i in range(0, len(position_list)):
        if if_inPoly(area, (position_list[i][0][0], position_list[i][0][1])) & if_inPoly(area, (
                position_list[i][1][0], position_list[i][1][1])) \
                & if_inPoly(area, (position_list[i][2][0], position_list[i][2][1])) & if_inPoly(area, (
                position_list[i][3][0], position_list[i][3][1])):
            selected_list.append(word_list[i])
    return selected_list


def cut_name(s):
    person_list = ['丁志宇', '傅广卷', '郭洪雨', '郑永卫', '李伟平', '徐金胜', '吴锐', '郭昊亮', '陈新国', '包泮旺', '徐羊敏', '王丰平', '马芹纲', '吴根强',
                   '丁海洋', '郑军华', '高能', '周义程', '耿驰远']
    name = ''
    words = pseg.cut(s, use_paddle=True)
    for word, flag in words:
        for person in person_list:
            if difflib.SequenceMatcher(None, word, person).quick_ratio() >= 0.5:
                return person
    # for word, flag in words:
    #     if difflib.SequenceMatcher(None, word, '海洋').quick_ratio() >= 0.6:
    #         return '丁海洋'
    #     if flag == 'PER' or flag == 'nr':
    #         name = word
    #         return name
    return name


# for i in picList:
#     # 解决图片路径中中文问题
#     img = cv2.imdecode(np.fromfile(picPath + i, dtype=np.uint8), -1)
#     # 读取图像
#     result = reader.readtext(img, canvas_size=4096)
#
#     word_list = []
#     position_list = []
#     sketch_list = []
#     table_list = []
#
#     for j in result:
#         position = j[0]
#         word = j[1]
#         recognitionRate = j[2]
#         position_list.append(position)
#         word_list.append(word)
#         print(j)
#         if '示意' in word or '大样' in word or '设计图' in word or '截面' in word or '剖面' in word or '平面' in word or '立面' in word or '配置图' in word or '布置图' in word:
#             if len(word) <= 18:
#                 sketch_list.append(word)
#
#         if '数量表' in word or '尺寸表' in word or '材料表' in word:
#             table_list.append(word)
#
#     department = keyword_start(word_list, position_list, '浙江省交通规划设计研究院') if keyword_start(word_list, position_list,
#                                                                                           '浙江省交通规划设计研究院') is not None else [
#         [640, 6383], [3339, 6383], [3339, 6661], [640, 6661]]
#     # 提取图纸名字
#     left_bottom_x = department[0][0] + 2480
#     left_bottom_y = department[1][1] - 50
#     right_top_x = left_bottom_x + 1900
#     right_top_y = left_bottom_y + 500
#     # print(left_bottom_x, left_bottom_y, right_top_x, right_top_y)
#     drawing_name = ''.join(
#         get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list,
#                                word_list))
#
#     # 提取设计者
#     left_bottom_x = department[0][0] + 4180
#     left_bottom_y = department[1][1] - 150
#     right_top_x = left_bottom_x + 1300
#     right_top_y = left_bottom_y + 500
#     designer = cut_name(''.join(
#         get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list)))
#
#     # 提取复核者
#     left_bottom_x = department[0][0] + 5320
#     left_bottom_y = department[1][1] - 150
#     right_top_x = left_bottom_x + 1300
#     right_top_y = left_bottom_y + 500
#     reviewer = cut_name(''.join(
#         get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list)))
#
#     # 提取审核者
#     left_bottom_x = department[0][0] + 6250
#     left_bottom_y = department[1][1] - 150
#     right_top_x = left_bottom_x + 2000
#     right_top_y = left_bottom_y + 500
#     viewer = cut_name(''.join(
#         get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list)))
#
#     # 提取图号
#     left_bottom_x = department[0][0] + 8100
#     left_bottom_y = department[1][1] - 150
#     right_top_x = left_bottom_x + 900
#     right_top_y = left_bottom_y + 500
#     drawing_num = ' '.join(
#         get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list)).strip(
#         '图号')
#
#     note = keyword_start(word_list, position_list, '注') if keyword_start(word_list, position_list,
#                                                                          '注') is not None else keyword_start(
#         word_list, position_list, '说明')
#     if note is not None:
#         # 提取注解或说明
#         left_bottom_x = note[0][0] - 50
#         left_bottom_y = note[0][1] - 50
#         right_top_x = left_bottom_x + 3300
#         right_top_y = left_bottom_y + 1800
#         annotation = ''.join(
#             get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list))
#     else:
#         annotation = ''
#
#     sketch = ','.join(sketch_list)
#     sheet = ','.join(table_list)
#     # 如果图号在注释中被识别
#     if len(annotation) != 0 and '图号' in annotation:
#         s = annotation.split('图号', 1)
#         annotation = s[0]
#         if len(drawing_num) == 0:
#             drawing_num = s[1]
#
#     if '审核' in annotation or '复核' in annotation or viewer in annotation or reviewer in annotation:
#         annotation = annotation.strip('审核').strip('复核').strip(viewer).strip(reviewer)
#
#     eachList = [i, drawing_name, drawing_num, designer, viewer, reviewer, annotation, sketch, sheet]
#     print(eachList)
#     cadList.append(eachList)

# columnsName = ['图片名字', '图纸名字', '图号', '设计者', '审核者', '复核者', '说明注释', '包含图示', '包含表格']
# pd.DataFrame(columns=columnsName, data=cadList).to_csv('data/cad_pic_text.csv', encoding='utf-8')
#
# label_list = ['公路工程', '施工图设计', '隧道']
#
# result_list = []
#
# with open("record.json", "w", newline="", encoding="utf-8") as f:
#     for i in cadList:
#         cad_dict = {
#             'drawing_name': i[1],
#             'drawing_num': i[2],
#             'designer': i[3],
#             'viewer': i[4],
#             'reviewer': i[5],
#             'annotation': i[6],
#             'sketch': i[7],
#             'sheet': i[8],
#             'label': label_list
#         }
#         result_list.append(cad_dict)
#     json.dump(result_list, f, ensure_ascii=False)
# print("加载入文件完成...")

if __name__ == '__main__':
    picPath = 'img/'
    picList = os.listdir(picPath)

    cadList = []

    # 创建reader对象
    reader = easyocr.Reader(['ch_sim', 'en'])

    # 解决图片路径中中文问题
    img = cv2.imdecode(np.fromfile(picPath + '39.png', dtype=np.uint8), -1)
    # 读取图像
    result = reader.readtext(img, canvas_size=4096)

    word_list = []
    position_list = []
    sketch_list = []
    table_list = []

    for j in result:
        position = j[0]
        word = j[1]
        recognitionRate = j[2]
        position_list.append(position)
        word_list.append(word)
        print(j)
        if '示意' in word or '大样' in word or '设计图' in word or '截面' in word or '剖面' in word or '平面' in word or '立面' in word or '配置图' in word or '布置图' in word:
            sketch_list.append(word)

        if '数量表' in word or '尺寸表' in word or '材料表' in word:
            table_list.append(word)

    department = keyword_start(word_list, position_list, '浙江省交通规划设计研究院') if keyword_start(word_list, position_list, '浙江省交通规划设计研究院') is not None else [[407, 3183], [986, 3183], [986, 3256], [407, 3256]]

    # 提取项目名字
    left_bottom_x = department[1][0] - 50
    left_bottom_y = department[1][1] - 50
    right_top_x = left_bottom_x + 700
    right_top_y = left_bottom_y + 150
    # print(left_bottom_x, left_bottom_y, right_top_x, right_top_y)
    project_name = ''.join(
        get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list,
                               word_list))

    # 提取图纸名字
    left_bottom_x = department[0][0] + 1200
    left_bottom_y = department[1][1] - 50
    right_top_x = left_bottom_x + 800
    right_top_y = left_bottom_y + 150
    # print(left_bottom_x, left_bottom_y, right_top_x, right_top_y)
    drawing_name = ''.join(
        get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list,
                               word_list))

    # 提取设计者
    left_bottom_x = department[0][0] + 2100
    left_bottom_y = department[1][1] - 50
    right_top_x = left_bottom_x + 550
    right_top_y = left_bottom_y + 150
    designer = cut_name(''.join(
        get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list)))

    # 提取复核者
    left_bottom_x = department[0][0] + 2600
    left_bottom_y = department[1][1] - 50
    right_top_x = left_bottom_x + 550
    right_top_y = left_bottom_y + 150
    reviewer = cut_name(''.join(
        get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list)))

    # 提取审核者
    left_bottom_x = department[0][0] + 3150
    left_bottom_y = department[1][1] - 50
    right_top_x = left_bottom_x + 550
    right_top_y = left_bottom_y + 150
    viewer = cut_name(''.join(
        get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list)))

    # 提取图号
    left_bottom_x = department[0][0] + 3750
    left_bottom_y = department[1][1] - 50
    right_top_x = left_bottom_x + 550
    right_top_y = left_bottom_y + 150
    drawing_num = ' '.join(
        get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list)).strip(
        '图号')

    note = keyword_start(word_list, position_list, '注') if keyword_start(word_list, position_list,
                                                                         '注') is not None else keyword_start(
        word_list, position_list, '说明')
    if note is not None:
        # 提取注解或说明
        left_bottom_x = note[0][0] - 50
        left_bottom_y = note[0][1] - 50
        right_top_x = left_bottom_x + 1800
        right_top_y = left_bottom_y + 800
        annotation = ''.join(
            get_selected_area_word(left_bottom_x, left_bottom_y, right_top_x, right_top_y, position_list, word_list))
    else:
        annotation = ''

    sketch = ','.join(sketch_list)
    sheet = ','.join(table_list)
    # 如果图号在注释中被识别
    if len(annotation) != 0 and '图号' in annotation:
        s = annotation.split('图号', 1)
        annotation = s[0]
        if len(drawing_num) == 0:
            drawing_num = s[1]

    if '审核' in annotation or '复核' in annotation or viewer in annotation or reviewer in annotation:
        annotation = annotation.strip('审核').strip('复核').strip(viewer).strip(reviewer)

    eachList = [drawing_name, project_name, drawing_num, designer, viewer, reviewer, annotation, sketch, sheet]
    print(eachList)
