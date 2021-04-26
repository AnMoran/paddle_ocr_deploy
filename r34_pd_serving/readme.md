nvidia-docker build -t paddle_ocr:1.0.1 .

nvidia-docker run --name paddle_serving_ocr -d -it -p 9293:9293 -v /home/users/wangpengyuan/pd_serving/log:/log paddle_ocr:1.0.0 

nvidia-docker run --name paddle_serving_ocr -d -it -p 9293:9293 -p 9292:9292 -v /opt/webapps/develop/wangpengyuan/general_ocr/logs:/log -v /opt/webapps/develop/wangpengyuan/general_ocr/model:/model 139.224.123.246/paddlepaddle/serving:0.3.2-gpu

#修改镜像标签
docker tag paddle_ocr:1.0.0 192.168.102.96/paddlepaddle/serving:0.3.2-gpu

#上传镜像到私有库
docker push  192.168.102.96/paddlepaddle/serving:0.3.2-gpu

#另外一台机器上下载
docker pull 192.168.102.96/paddlepaddle/serving:0.3.2-gpu

#!/bin/bash
python -m paddle_serving_server_gpu.serve --model ./model/det_infer_server --port 9295 --gpu_id 0
python -m paddle_serving_server_gpu.serve --model ./model/rec_infer_server --port 9296 --gpu_id 1
