#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： LiuXG
# datetime： 2021/9/14 16:39 
# ide： PyCharm

tags = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
label_O = 17


def find_tag(out_path, B_label_id=1, I_label_id=2):
    """
    找到指定的label
    :param out_path: 模型预测输出的路径 shape = [1, rel_seq_len]
    :param B_label_id:
    :param I_label_id:
    :return:
    """
    sentence_tag = []
    for num in range(len(out_path)):
        if out_path[num] == B_label_id:
            start_pos = num
        if out_path[num] == I_label_id and out_path[num - 1] == B_label_id:
            length = 2
            for num2 in range(num, len(out_path)):
                if out_path[num2] == I_label_id and out_path[num2 - 1] == I_label_id:
                    length += 1
                    if num2 == len(out_path) - 1:  # 如果已经到达了句子末尾
                        sentence_tag.append((start_pos, length))
                        return sentence_tag
                if out_path[num2] == label_O:
                    sentence_tag.append((start_pos, length))
                    break
    return sentence_tag


def find_all_tag(out_path):
    num = 1  # 1: para、 2: noun
    result = {}
    for tag in tags:
        res = find_tag(out_path, B_label_id=tag[0], I_label_id=tag[1])
        result[num] = res
        num += 1
    return result