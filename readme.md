# YOLO人物检测

基于YOLOv8视频人物检测实例，python版本3.9.7 supervision版本0.23.0

![YOLO人物检测](https://github.com/rhfu/yolo/blob/main/video/yolo_1.gif)

- python=3.9.7
- supervision==0.23.0
- yolov8s.pt
- 其它依赖见 requirements.txt

## 安装

- git clone https://github.com/rhfu/yolo
- cd golo/
- conda create --name yolo python=3.9.7 && conda activate yolo
- pip install -r requirements.txt
- python main.py
  
## 目录说明

- video 为视频原文件及识别后文件
- model 为模型文件
- test 测试文件

## 附

- supervision官方文档
- https://supervision.roboflow.com/0.23.0/detection/annotators/
- yolov8 模型下载
- https://docs.ultralytics.com/tasks/detect/#models
