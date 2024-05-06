# encoding:utf-8

import requests
import base64

API_KEY = 'lyuUDvHD3VcjO0L8EAFRD40n'
# API_KEY = 'GMIWtKsFv8PYKOBtkvnlBPfP'

SECRET_KEY = 'eAK7S5MkBVYh3iNfWQDNX0FDFTE1x9WT'
# SECRET_KEY = 'HkGI88mDhh215v2w3B9lp5I2KiUXNBDq'

OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"


def read_file(image_path):
    f = None
    try:
        f = open(image_path, 'rb')
        return f.read()
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()


def fetch_token():
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + API_KEY + '&client_secret=' + SECRET_KEY
    response = requests.get(host)
    if response:
        result = response.json()
        if 'access_token' in result.keys() and 'scope' in result.keys():
            if not 'brain_all_scope' in result['scope'].split(' '):
                print('please ensure has check the  ability')
                exit()
            return result['access_token']
        else:
            print('please overwrite the correct API_KEY and SECRET_KEY')
            exit()


def ocr(token, img):
    texts = []
    locations = []
    # 读取测试图片
    base64_img = base64.b64encode(read_file(img))
    params = {"image": base64_img}
    request_url = OCR_URL + "?access_token=" + token
    headers = {'content-type': 'application/x-www-form-urlencoded'}

    try:
        requests.adapters.DEFAULT_RETRIES = 5  # 增加重连次数
        s = requests.session()
        s.keep_alive = False  # 关闭多余连接
        result = s.post(request_url, data=params, headers=headers)
        # 解析返回结果
        result_json = result.json()

        if 'error_code' in result_json.keys() or 'error_msg' in result_json.keys():
            print('ocr error.........')
            print(result_json)
            return None

        for words_result in result_json["words_result"]:
            texts.append(words_result["words"])
            locations.append(words_result["location"])

        return texts, locations

    except Exception as e:
        # import traceback
        # traceback.print_stack()
        print(e)
        return None
