from ultralytics import YOLO


class YOLODetector:
    def __init__(
        self,
        object_model_path="yolov8m.pt",
        robot_model_path="best.pt",
        conf=0.3,
        iou=0.5,
        debug=False
    ):
        """
        object_model_path : COCO model (general objects)
        robot_model_path  : finetuned robot-only model
        """

        # 🔹 Load both models
        self.object_model = YOLO(object_model_path)
        self.robot_model = YOLO(robot_model_path)

        self.conf = conf
        self.iou = iou
        self.debug = debug

        # General object classes you care about
        self.target_classes = [
            "person",
            "chair",
            "dining table",
            "tv",
            "laptop",
            "bottle",
            "glass"
        ]

    def detect(self, frame):

        detections = []

        # ===============================
        # 🔵 1. OBJECT DETECTION MODEL
        # ===============================
        object_results = self.object_model(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )

        for r in object_results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                class_name = self.object_model.names[cls_id]

                if class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detection = {
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": round(confidence, 2),
                    "class": class_name,
                    "source": "object_model"
                }

                detections.append(detection)

        # ===============================
        # 🔴 2. ROBOT DETECTION MODEL
        # ===============================
        robot_results = self.robot_model(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )

        for r in robot_results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(int, xyxy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detection = {
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": round(confidence, 2),
                    "class": "robot",  # force robot label
                    "source": "robot_model"
                }

                detections.append(detection)

        return detections