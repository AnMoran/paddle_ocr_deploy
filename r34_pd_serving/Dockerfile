#FROM hub.baidubce.com/paddlepaddle/serving:latest-cuda10.0-cudnn7
# 我编译时求快使用了自己的paddle镜像,正常应使用百度官方的镜像
FROM 139.224.123.246/paddlepaddle/paddle_env:1.0.0

#author
MAINTAINER wangpengyuan

RUN mkdir -p /app

WORKDIR /app

COPY requirements.txt /app

#使用python3.6环境
RUN pip3.6 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

COPY . /app

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

