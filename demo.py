#coding=utf-8

import os
import time
import requests
import json
import base64
import sys


def main():
    
    #测试 imgFile

    fp = open(sys.argv[1],'rb')
    imgbuf = fp.read()
    fp.close()

    files = {'imgFile':  imgbuf}
    print('testing ....')
    start = time.time()
    data = {"appId":"000000"}
    res=requests.post(url="http://0.0.0.0:8101/v2/general_ocr/",data=data, files=files)
    end = time.time()
    result = json.loads(res.text)
    json_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(json_str)
    time_cost = round(end-start,3)
    print('time_cost = %s seconds\n'%time_cost) 

    #测试imgUrl

    # data = {"appId":"000000","imgUrl":"http://ceshiweb.aibaoxian.com/ai/static2/demo_img/general_ocr/02.jpg"}
    # print('testing ....')
    # start = time.time()
    # res=requests.post(url="https://api.aibao.com/general_ocr/v2/general_ocr/",data=data)
    # end = time.time()
    # result = json.loads(res.text)
    # json_str = json.dumps(result, indent=4, ensure_ascii=False)
    # print(json_str)  
    # time_cost = round(end-start,3)
    # print('time_cost = %s seconds\n'%time_cost) 

    #测试imgBase64

    fp = open(sys.argv[1],'rb')
    imgbuf = fp.read()
    fp.close()
    print('testing ....')
    start = time.time()
    data = {
            "appId":"000000",
            "imgBase64":base64.b64encode(imgbuf)
            }
    res=requests.post(url="http://0.0.0.0:8101/v2/general_ocr/",data=data)
    end = time.time()
    result = json.loads(res.text)
    json_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(json_str)    
    time_cost = round(end-start,3)
    print('time_cost = %s seconds\n'%time_cost)

if __name__ == "__main__":
    main()
