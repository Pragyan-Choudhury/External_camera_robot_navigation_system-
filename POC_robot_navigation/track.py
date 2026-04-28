import math


class Tracker:
    def __init__(self, iou_threshold=0.3, max_lost=10):
        """
        iou_threshold : minimum IOU required to match detection to track
        max_lost      : number of frames to keep lost tracks
        """
        self.next_id = 0
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    # 🔹 Generate unique ID
    def _get_new_id(self):
        self.next_id += 1
        return self.next_id

    # 🔹 Compute IOU between two bounding boxes
    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        inter_area = inter_width * inter_height

        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union_area = boxA_area + boxB_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    # 🔥 Main update function
    def update(self, detections):
        """
        detections: list of dicts from YOLO
                    each dict must contain:
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class": class_name
                    }
        """

        updated_tracks = []
        used_track_ids = set()

        # 🔹 Match detections to existing tracks
        for det in detections:

            bbox = det["bbox"]
            cls = det["class"]

            best_iou = 0
            best_track = None

            for track in self.tracks:

                # Match only same class
                if track["class"] != cls:
                    continue

                iou_score = self._iou(bbox, track["bbox"])

                if iou_score > best_iou:
                    best_iou = iou_score
                    best_track = track

            # 🔹 If good match found
            if best_iou > self.iou_threshold and best_track is not None:

                best_track["bbox"] = bbox
                best_track["lost"] = 0

                updated_tracks.append(best_track)
                used_track_ids.add(best_track["id"])

            else:
                # 🔹 Create new track
                new_track = {
                    "id": self._get_new_id(),
                    "bbox": bbox,
                    "class": cls,
                    "lost": 0
                }

                updated_tracks.append(new_track)
                used_track_ids.add(new_track["id"])

        # 🔹 Handle lost tracks
        for track in self.tracks:
            if track["id"] not in used_track_ids:

                track["lost"] += 1

                if track["lost"] <= self.max_lost:
                    updated_tracks.append(track)

        self.tracks = updated_tracks

        # 🔥 Final output with CENTER added
        output_tracks = []

        for track in self.tracks:

            x1, y1, x2, y2 = track["bbox"]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            output_tracks.append({
                "id": track["id"],
                "bbox": track["bbox"],
                "center": [cx, cy],   # ✅ Added center
                "class": track["class"]
            })

        return output_tracks