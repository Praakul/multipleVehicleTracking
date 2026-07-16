import numpy as np
from kalman import KalmanFilter

class Track:
    def __init__(self, track_id, initial_bbox):
        self.track_id = track_id
        self.kf = KalmanFilter()
        
        # Initialize the filter with the first detection
        # We convert [x1, y1, x2, y2] to [cx, cy, w, h]
        self.kf.state[:4] = self.bbox_to_state(initial_bbox)
        

        self.bbox = initial_bbox
        
        self.time_since_update = 0 
        self.hits = 1 

    def bbox_to_state(self, bbox):
        """Converts [x1, y1, x2, y2] to [center_x, center_y, scale, ratio]"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        scale = width * height
        ratio = width / height if height > 0 else 0
        return np.array([center_x, center_y, scale, ratio])

    def state_to_bbox(self, state):
        """Converts [center_x, center_y, scale, ratio, ...] to [x1, y1, x2, y2]"""
        cx, cy, s, r = state[:4]
        # s = w * h and r = w / h
        # w = sqrt(s * r)
        # h = s / w or sqrt(s / r)
        
        # To avoid math domain errors due to numerical instability:
        s = max(0, s)
        r = max(1e-6, r)
        
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]

    def predict(self):
        """Predict the next state using the Kalman Filter."""
        predicted_state = self.kf.predict()
        
        self.bbox = self.state_to_bbox(predicted_state)
        self.time_since_update += 1

    def update(self, bbox):
        """Update the Kalman Filter with a new detection."""
        measurement = self.bbox_to_state(bbox)
        
        self.kf.update(measurement)
        self.bbox = self.state_to_bbox(self.kf.state)
        self.time_since_update = 0
        self.hits += 1