# -*- coding: utf-8 -*-

from PyPDF2 import PdfFileReader, PdfFileWriter
import os
import fitz
import ocr

pdf_path = "pdf/"
img_path = "dataset/train/img/"

print("*" * 30 + "开始拆分PDF" + "*" * 30)
file_reader = PdfFileReader("train.pdf")
total_pages = file_reader.getNumPages()
for page in range(total_pages):
    # 实例化对象
    file_writer = PdfFileWriter()
    # 将遍历的每一页添加到实例化对象中
    file_writer.addPage(file_reader.getPage(page))
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)
    with open("pdf/{}.pdf".format(page), 'wb') as out:
        file_writer.write(out)

pdfList = os.listdir(pdf_path)
pdfList.sort(key=lambda x: int(x.split('.')[0]))
print("*" * 30 + "开始转换图片" + "*" * 30)
for p in pdfList:
    pdf = fitz.open(pdf_path + p)
    page = pdf[0]
    trans = fitz.Matrix(4, 4).prerotate(0)
    pm = page.get_pixmap(matrix=trans, alpha=False)
    # 开始写图像
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    pm.save(img_path + 'train_' + p.split(".")[0] + ".png")
    pdf.close()

# print("*" * 30 + "开始OCR识别" + "*" * 30)
# # 创建reader对象
# reader = easyocr.Reader(['ch_sim', 'en'])
#
# ocr.ocr_single(reader, img_path, 4096)
# ocr.ocr_full(reader, img_path, 4096)
# with open("word.txt", "r", encoding="utf-8") as f:
#     drawing_words = f.read()
# ocr.text_analyze(drawing_words, 0, 50)
# ocr.text_analyze(drawing_words, 1, 50)
