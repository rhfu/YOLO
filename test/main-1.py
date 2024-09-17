import numpy as np
import supervision as sv
from ultralytics import YOLO

class CountObject():

    def __init__(self, input_video_path, output_video_path) -> None:
        # 加载YOLOv8模型
        self.model = YOLO('model/yolov8s.pt')

        # 输入视频, 输出视频
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # 视频信息
        self.video_info = sv.VideoInfo.from_video_path(input_video_path)

        # 检测框属性
        self.box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=2, text_scale=2)

    def process_frame(self, frame: np.ndarray, _) -> np.ndarray:
        # 检测
        results = self.model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_yolov8(results)
        detections = detections[detections.class_id == 0]

        # 绘制检测框
        # box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        labels = [f"{self.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame

    def process_video(self):
        # 处理视频
        sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path,
                         callback=self.process_frame)


if __name__ == "__main__":
    video = "3.mp4"
    obj = CountObject(f"input/{video}", f"output/{video}")
    obj.process_video()