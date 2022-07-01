from typing import Union
from fastapi import FastAPI, UploadFile
import search
import baidu_extract
from config.config import config
import utils

app = FastAPI(title='图纸管理系统', description='接口文档')


@app.post("/upload", summary='上传文件接口')
async def upload_file(file: UploadFile):
    content = await file.read()
    utils.init_dirs(config.temp_path)
    file_path = config.temp_path + file.filename
    with open(file_path, 'wb') as f:
        f.write(content)
    result = baidu_extract.extract(file_path)
    return result


@app.get("/search", summary='查询接口')
def search_drawing(query: str):
    result = search.search("drawing", query)
    return result


@app.get("/wordcloud", summary='获取词云接口')
def request_wordcloud():
    result = search.wordcloud('drawing')
    return result
