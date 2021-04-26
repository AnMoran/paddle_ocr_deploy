# paddle_ocr_deploy
paddle_ocr的paddle serving部署，支持gRPC

一、r34_pd_serving为编译生成serving端镜像的目录

cd r34_pd_serving
docker build -t paddle_ocr:0.0.0 .

# 启动服务端

docker-compose -f docker-compose-serving.yml up -d
# 经过一段漫长时间的下载二进制c++运算库，可以在映射目录下看到det.log和rec.log

# 将容器保存为镜像，方便下次启动时不再下载二进制c++文件，能瞬间启动(二进制文件有点大)
sudo docker commit paddle_ocr paddle_ocr:1.0.0

二、r34_pd_client为编译生成client端镜像的目录

cd r34_pd_client
docker build -t general_ocr:1.0.0 .

# 启动客户端
docker-compose -f docker-compose-client.yml up -d

# 测试是否可用
python demo.py
