import os
import sys
import importlib
importlib.reload(sys)

#from io import StringIO
#from PIL import Image
#--------------------------------------------------------------------------------------------------------------------------------
from flask import Flask
from flask import request,jsonify
import requests
import time
from general_ocr_recog import general_ocr_client
from paddle_serving_client import Client
import random
import json
import time
import datetime
import numpy as np
import cv2
from  mylog import MyLog
logger = MyLog('service').getlog()
from service_config import *
#--------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

medical_image_folder = '/data/images/'
medical_json_folder = '/data/json/'

#初始化检测和识别
det_client = Client()
det_client.load_client_config("./general_ocr_config/det_infer_client/serving_client_conf.prototxt")
det_client.connect(det_ip_port)

#start rec Client
rec_client = Client()
rec_client.load_client_config("./general_ocr_config/rec_infer_client/serving_client_conf.prototxt")
rec_client.connect(rec_ip_port)

@app.route("/v2/general_ocr/", methods=["POST"])
def predict_medical():
    appId = request.form.get("appId")
    img_file = request.files.get("img_file")
    img_url = request.form.get("img_url")

    if appId is None:
        results = {'status': -1111, 'err_msg':'param error', 'medical_result':None}
        json_results = json.dumps(results,ensure_ascii=False)
        logger.info(json_results)
        return jsonify(results)

    logger.info('appId=%s'%appId)

    if img_file is not None:
        imgbuf = img_file.stream.read()
        imgbuf_len = len(imgbuf) 
    elif img_url is not None:
        try:
            response = requests.request('GET', img_url, timeout=5)
        except Exception as e:
            results = {'status': -1114, 'err_msg':'image download error', 'medical_result':None}
            json_results = json.dumps(results,ensure_ascii=False)
            logger.info(json_results)
            return jsonify(results)
        else:
            if 200 == response.status_code:
                imgbuf = response.content
                imgbuf_len = len(imgbuf)
            else:
                results = {'status': -1114, 'err_msg':'image download error', 'medical_result':None}
                json_results = json.dumps(results,ensure_ascii=False)
                logger.info(json_results)
                return jsonify(results)
    else:
        results = {'status': -1111, 'err_msg':'param error', 'medical_result':None}
        json_results = json.dumps(results,ensure_ascii=False)
        logger.info(json_results)
        return jsonify(results)

    logger.info('image buf length = %s'%imgbuf_len)

    if imgbuf_len > 2*1024*1024:
        results = {'status': -1112, 'err_msg':'image buf length exceeds 2MB', 'medical_result':None}
        json_results = json.dumps(results,ensure_ascii=False)
        logger.info(json_results)
        return jsonify(results)

    timestamp = int(round(time.time() * 1000))
    img_name_0 = '%s_%s'%(appId,timestamp)
    img_name = img_name_0 + '.jpg'

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    todaydir = medical_image_folder + today + '/'
    if not os.path.exists(todaydir):
        os.mkdir(todaydir)

    filepath =  os.path.join(todaydir, img_name)
    fpOut = open(filepath,'wb')
    fpOut.write(imgbuf)
    fpOut.close()

    simage = np.asarray(bytearray(imgbuf), dtype='uint8')
    try:
        image = cv2.imdecode(simage, cv2.IMREAD_COLOR)
    except Exception as e:
        results = {'status': -1113, 'err_msg':'image format error', 'text_result':None}    
        json_results = json.dumps(results,ensure_ascii=False)
        logger.info(json_results)
        return jsonify(results)
    if image is None:
        results = {'status': -1113, 'err_msg':'image format error', 'text_result':None}
        json_results = json.dumps(results,ensure_ascii=False)
        logger.info(json_results)
        return jsonify(results)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        image = np.tile(image, (1, 1, 3))
    elif image.ndim == 3 and image.shape[(-1)] == 4:
        image = image[:, :, :3]
  
    try:
        # cardresult = process_drivecards(image,server)
        boxes_list,text_list,score_list = general_ocr_client(image,det_client,rec_client)
    except Exception as e:
        results = {'status': -200, 'err_msg':'image recog service interal error', 'text_result':None}
        json_results = json.dumps(results,ensure_ascii=False)
        logger.info(json_results)
        return jsonify(results)
    
    if len(boxes_list) == 0:
        results = {'status': -100, 'err_msg':'text not detected in image','text_result':None}
        json_results = json.dumps(results,ensure_ascii=False)
        logger.info(json_results)
        return jsonify(results)

        
    results = {}
    results['status'] = 0
    results['err_msg'] = ''
    medical_result = {}
    #  medical_result = {'index':index,'text_line_dict':text_line_dict
    #                      '''
    #                   }
    # text_line_dict = {'box':'x1,y1,x2,y2,x3,y3,x4,y4','text':'text_result','confidence':'confidence'}
    

    for index in range(len(boxes_list)):
        text_line_dict = {}
        box = boxes_list[index]
        box_value = str(box[0][0]) + ',' +str(box[0][1]) + ',' +str(box[1][0]) + ',' +str(box[1][1]) + ',' + str(box[2][0]) + ',' +str(box[2][1]) + ',' +str(box[3][0]) + ',' +str(box[3][1])
        text_line_dict['box'] = box_value
        text_line_dict['text'] = text_list[index]
        text_line_dict['confidence'] = str(score_list[index])
        medical_result[str(index)] = text_line_dict
    results['medical_result'] =   medical_result  
    #保存识别结果为json文件
    json_todaydir = medical_json_folder + today + '/'
    if not os.path.exists(json_todaydir):
        os.mkdir(json_todaydir)    
    
    json_name = img_name[:-4]+'.json'
    json_filepath = os.path.join(json_todaydir,json_name)

    json_results = json.dumps(results,indent=4,ensure_ascii=False)
    # logger.info(json_results)
    with open(json_filepath, "w+",encoding='utf-8') as f:
        f.write(json_results)
    print("recognition success!")
    return jsonify(results)




if __name__ == "__main__":
    app.run(host=flask_host, port=flask_port, debug=True, use_reloader=False)
