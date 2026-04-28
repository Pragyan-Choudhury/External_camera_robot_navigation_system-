from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8m.pt", conf=0.2, iou=0.5, debug=False):
        """
        model_path: YOLO model (yolov8n, yolov8m, yolov8l)
        conf: confidence threshold
        iou: NMS IoU threshold
        debug: print all detected classes
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.debug = debug

        # Classes you care about
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
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)

        detections = []
        bottle_candidates = []

        for r in results:
            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                class_name = self.model.names[cls_id]

                # 🔹 Debug print
                if self.debug:
                    print(f"[DEBUG] Detected: {class_name} ({confidence:.2f})")

                # 🔹 Filter only required classes
                if class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)

                # 🔹 Compute center (important for navigation)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detection = {
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": round(confidence, 2),
                    "class": class_name,
                    "id": None  # for future tracking
                }

                # 🔹 Separate bottles
                if class_name == "bottle":
                    bottle_candidates.append(detection)
                else:
                    detections.append(detection)

        # 🔥 Identify robot (highest confidence bottle)
        if bottle_candidates:
            best_bottle = max(bottle_candidates, key=lambda x: x["confidence"])

            for bottle in bottle_candidates:
                if bottle is best_bottle:
                    bottle["class"] = "robot"
                else:
                    bottle["class"] = "bottle"

                detections.append(bottle)

        return detections