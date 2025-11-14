# File: track.py

import numpy as np
from kalman import KalmanFilter

class Track:
    def __init__(self, track_id, initial_bbox):
        self.track_id = track_id
        self.kf = KalmanFilter()
        
        # Initialize the filter with the first detection
        # We convert [x1, y1, x2, y2] to [cx, cy, w, h]
        self.kf.state[:4] = self.bbox_to_state(initial_bbox)
        
        # This is the *last known* bounding box
        self.bbox = initial_bbox
        
        self.time_since_update = 0 # Frames since last detection
        self.hits = 1 # Number of times this track has been detected

    def bbox_to_state(self, bbox):
        """Converts [x1, y1, x2, y2] to [center_x, center_y, width, height]"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        return np.array([center_x, center_y, width, height])

    def state_to_bbox(self, state):
        """Converts [center_x, center_y, width, height, ...] to [x1, y1, x2, y2]"""
        cx, cy, w, h = state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]

    def predict(self):
        """Predict the next state using the Kalman Filter."""
        predicted_state = self.kf.predict()
        
        # Convert predicted state back to a bounding box
        self.bbox = self.state_to_bbox(predicted_state)
        self.time_since_update += 1

    def update(self, bbox):
        """Update the Kalman Filter with a new detection."""
        # Convert measurement
        measurement = self.bbox_to_state(bbox)
        
        # Update the filter
        self.kf.update(measurement)
        
        # Update track properties
        # The state from the KF is now the "ground truth"
        self.bbox = self.state_to_bbox(self.kf.state)
        self.time_since_update = 0
        self.hits += 1