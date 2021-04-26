#编译镜像
docker build -t medical_ocr:1.0.0 .
#启动服务
nvidia-docker run --name medical_ocr -d -it --network paddle_net -p 9294:9294 -v /home/users/wangpengyuan/pd_client/log:/app/log medical_ocr:1.0.0 

python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.3.2-cp36-none-any.whl https://paddle-serving.bj.bcebos.com/whl/paddle_serving_app-0.1.2-py3-none-any.whl