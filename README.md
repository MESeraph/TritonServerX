# TritonServerX

## 00 | 准备运行环境
1. 根据本机NVIDIA环境pull相应的docker镜像，具体参考[support-matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
注：我用的宿主机环境是CUDA 11.4，Driver 170+，故选择21.07版本。
* docker pull nvcr.io/nvidia/tritonserver:21.07-py3-sdk
* docker pull nvcr.io/nvidia/tritonserver:21.07-py3
* docker pull nvcr.io/nvidia/pytorch:21.07-py3  (用于yolov4的模型转换)

2. 将TritonServerX文件夹拷贝到工作目录

## 01 | 转换模型  
选择你需要使用的模型转换即可
### yolov4模型转换
1. 使用如下命令run docker：   
```sh
docker run --gpus all -itd --network host --name pytorch-yolov4 -v /path/TritonServerX:/opt/workspace nvcr.io/nvidia/pytorch:21.07-py3 bash
```
2. 参考[pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)，将模型按weights->onnx->tensorrt过程进行转换:   

* weights->onnx:    
    (1) 修改cfg/yolov4.cfg中net字段中的width、height为512，执行如下命令:   
    ```sh
    python demo_darknet2onnx.py cfg/yolov4.cfg /opt/workspace/labels/coco.names ./yolov4.weights data/dog.jpg -1
    ``` 
    (2) 将生产后的yolov4_-1_3_512_512_dynamic.onnx拷贝至server/yolov4/yolov4/yolov4_onnx/1目录下，并修改文件名位model.onnx     

* onnx->tensorrt:   
    (1) 将上面生成的yolov4_-1_3_512_512_dynamic.onnx按如下命令执行：
    ```sh
    trtexec --onnx=yolov4_-1_3_512_512_dynamic.onnx --explicitBatch  --optShapes=input:4x3x512x512 --maxShapes=input:8x3x512x512 --minShapes=input:1x3x512x512 --saveEngine=yolov4_8_3_512_512.engine --fp16
    ```
    (2) 生成后的模型放至server/yolov4/yolov4_ensemble/yolov4_tensorrt/1和server/yolov4/yolov4_ensemble_uint8/yolov4_tensorrt/1目录下，并修改文件名位model.plan

## 02 | 启动triton server
1. 使用如下命令run docker：
```sh
docker run --gpus all -itd  -p8000:8000 -p8001:8001 -p8002:8002 --name triton-server -v /path/TritonServerX/server:/models --shm-size 5G nvcr.io/nvidia/tritonserver:21.07-py3 bash
```
2. 进入docker容器，输入一下命令启动triton server：
```sh
pip install pillow
tritonserver --model-repository /models/模型类型/模型目录
# eg: tritonserver --model-repository /models/yolov4/yolov4_ensemble
```
## 03 | 启动triton client并调用triton服务
1. 使用如下命令run docker：
```sh
docker run  -itd --network host --name triton-client -v /path/TritonServerX/client:/client nvcr.io/nvidia/tritonserver:21.07-py3-sdk bash
```
2. 进入docker容器，并进入`/client/模型类型/模型目录`，执行一下命令：
```sh
pip install opencv-python 
python ./client.py -m 模型服务名 ../../data/dog.jpg --out
#eg: python ./client.py -m ensemble_python_yolov4 ../data/dog.jpg --out
```
