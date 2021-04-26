#!/bin/bash
nohup python3 -m paddle_serving_server_gpu.serve --model /model/det_infer_server --port 9293 --gpu_id 0 >/log/det.log 2>&1 &
python3 -m paddle_serving_server_gpu.serve --model /model/rec_infer_server --port 9292 --gpu_id 1 >/log/rec.log 2>&1 
