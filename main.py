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
		# self.video_info = sv.VideoInfo.from_video_path(input_video_path)

		# box 检测框属性
		color_lookup = ["green", "yellow", "black", "blue", "red", "white", "orange"]
		# sv.ColorPalette.from_matplotlib('viridis', 5)
		self.box_annotator = sv.BoxAnnotator()
		
		# label检测标签属性
		# color-文字背景色，text_color-文字颜色，text_scale-文字大小
		# text_position-文字位置，text_thickness-文字粗细，text_padding-文字填充距离
		# self.label_annotator = sv.LabelAnnotator(color=sv.Color(r=128, g=0, b=128),text_color=sv.Color(r=255, g=255, b=255), text_scale=1, text_position=sv.Position.TOP_LEFT, text_thickness=2,text_padding=5)
		self.label_annotator = sv.LabelAnnotator()
		
		# trace
		self.tracker = sv.ByteTrack()

	def process_frame(self, frame: np.ndarray, _) -> np.ndarray:
		# 检测
		results = self.model(frame, imgsz=1280)[0]
		detections = sv.Detections.from_ultralytics(results)
		# 分类 https://cloud.tencent.com/developer/article/2392864
		selected_classes = [0, 2, 32]
		detections = detections[np.isin(detections.class_id, selected_classes)]
		detections = self.tracker.update_with_detections(detections)

		annotated_frame = frame.copy()
		
		# trace
		# annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
		# box
		annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
		
		# lables
		labels = [
			f"#{tracker_id} {class_name} {confidence:.2f}"
			for tracker_id, class_name, confidence
			in zip(detections.tracker_id, detections['class_name'], detections.confidence)
		]
		annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
		# sv.plot_image(annotated_frame, (12, 12))
		return annotated_frame

	def process_video(self):
		# 处理视频
		sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path,
						 callback=self.process_frame)


if __name__ == "__main__":
	video = "car.mp4"
	obj = CountObject(f"video/{video}", f"video/result_{video}")
	obj.process_video()
