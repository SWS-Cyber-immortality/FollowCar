import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
import cv2

class SORTObject:
    def __init__(self, bbox, track_id):
        self.bbox = np.array(bbox)
        self.track_id = track_id
        self.hits = 1
        self.age = 1

class SORT:
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.tracked_objects = deque()

    def update(self, detections):
        # Predict the motion of existing tracked objects
        for obj in self.tracked_objects:
            obj.bbox = self._predict(obj.bbox)

        # Data association using Hungarian Algorithm
        if len(self.tracked_objects) > 0 and len(detections) > 0:
            track_bboxes = np.array([obj.bbox for obj in self.tracked_objects])
            detection_bboxes = np.array([detection for detection in detections])
            iou_matrix = self._compute_iou_matrix(track_bboxes, detection_bboxes)
            track_indices, detection_indices = linear_sum_assignment(-iou_matrix)

            for i, j in zip(track_indices, detection_indices):
                if iou_matrix[i, j] > self.iou_threshold:
                    self.tracked_objects[i].bbox = detections[j]
                    self.tracked_objects[i].hits += 1
                    self.tracked_objects[i].age = 0
                else:
                    self._remove_track(i)

        # Create new tracks for unmatched detections
        for detection in detections:
            if not obj:continue
            if not any(self._iou(detection, obj.bbox) > self.iou_threshold for obj in self.tracked_objects):
                self._create_track(detection)

        # Update age of each track
        for obj in self.tracked_objects:
            obj.age += 1

        # Remove old tracks
        self._remove_old_tracks()

        # Return the detected object with the highest number of hits (most reliable)
        if len(self.tracked_objects) > 0:
            return max(self.tracked_objects, key=lambda obj: obj.hits)
        else:
            return None

    def _predict(self, bbox):
        # Implement motion prediction (e.g., Kalman filter or simple linear prediction)
        # For simplicity, this implementation just returns the same bbox without any prediction.
        return bbox

    def _compute_iou_matrix(self, bboxes1, bboxes2):
        # Compute IoU (Intersection over Union) matrix between two sets of bounding boxes
        iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))

        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                iou_matrix[i, j] = self._iou(bbox1, bbox2)

        return iou_matrix

    def _iou(self, bbox1, bbox2):
        # Compute IoU (Intersection over Union) between two bounding boxes
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area

    def _create_track(self, bbox):
        # Create a new tracked object for the unmatched detection
        track_id = self.next_track_id
        self.next_track_id += 1
        new_obj = SORTObject(bbox, track_id)
        self.tracked_objects.append(new_obj)

    def _remove_track(self, index):
        # Remove a track from the list of tracked objects
        self.tracked_objects[index] = None

    def _remove_old_tracks(self):
        # Remove tracks that have not been updated for a certain number of frames
        self.tracked_objects = deque([obj for obj in self.tracked_objects if obj is not None and obj.age <= self.max_age])

# Initialize YOLO detector and SORT tracker
# yolonet = yolov5(args.net_type, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)
# sort_tracker = SORT()  # Implement the SORT class based on the above cod

# while True:
#     # Read a frame from the camera
#     ret, srcimg = cap.read()
#     if not ret:
#         break

#     # Perform object detection on the frame using YOLO
#     dets = yolonet.detect(srcimg)
#     srcimg = yolonet.postprocess(srcimg, dets)

#     # Feed the detections to the SORT tracker for data association and tracking
#     tracked_object = sort_tracker.update(dets)

#     # Draw bounding box and track ID on the frame
#     if tracked_object is not None:
#         bbox = tracked_object.bbox.astype(int)
#         cv2.rectangle(srcimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2)
#         cv2.putText(srcimg, f"Track ID: {tracked_object.track_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
