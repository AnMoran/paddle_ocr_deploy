# FROM hub.baidubce.com/paddlepaddle/serving:latest-cuda10.0-cudnn7
# 我为了镜像大小均采用了自己编译出的基础镜像，采用上面那个百度的镜像也能成功就是image会大一点
FROM 139.224.123.246/aibao_ai_deploy/aibao_ai_deploy:1.6

#author
MAINTAINER wangpengyuan

RUN mkdir -p /app

WORKDIR /app

COPY . /app

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

