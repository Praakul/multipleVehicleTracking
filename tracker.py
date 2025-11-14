# File: tracker.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from track import Track
from kalman import iou

# --- Tunable Constants ---
# Min IoU to be considered a match. (1 - 0.7 = 0.3)
IOU_MATCH_THRESHOLD = 0.3 
# Max frames to keep a track without a new detection.
MAX_FRAMES_SINCE_UPDATE = 5 
# Min detections to "confirm" a track (not used here, but good for DeepSORT)
# MIN_HITS_TO_START = 3 
# -------------------------

class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_track_id = 0

    def update(self, detections):
        # 1. Predict the next state for all existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Match detections with tracks using the Hungarian algorithm
        if len(self.tracks) > 0 and len(detections) > 0:
            # Create a cost matrix based on IoU
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    # Cost is 1 - IoU (since linear_sum_assignment minimizes cost)
                    cost_matrix[i, j] = 1 - iou(track.bbox, det) 
            
            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 3. Update matched tracks, handle unmatched tracks/detections
            matched_indices = set()
            unmatched_track_indices = set(range(len(self.tracks)))

            for r, c in zip(row_ind, col_ind):
                # If the cost is less than the threshold (i.e., IoU is > threshold)
                if cost_matrix[r, c] < (1 - IOU_MATCH_THRESHOLD):
                    self.tracks[r].update(detections[c])
                    matched_indices.add(c)
                    unmatched_track_indices.remove(r)
            
            unmatched_detections = [det for i, det in enumerate(detections) if i not in matched_indices]
        
        else:
            # If no tracks or no detections, all detections are unmatched
            unmatched_detections = detections
            unmatched_track_indices = set(range(len(self.tracks)))
            
        # 4. Create new tracks for unmatched detections
        for det in unmatched_detections:
            new_track = Track(self.next_track_id, det)
            self.tracks.append(new_track)
            self.next_track_id += 1
            
        # 5. Remove old tracks that haven't been seen for a while
        # We also check that a track is not "just born" (e.g., hits > 1)
        # to avoid deleting tentative tracks, but for simple SORT, this is fine.
        self.tracks = [t for i, t in enumerate(self.tracks) if i not in unmatched_track_indices or t.time_since_update <= MAX_FRAMES_SINCE_UPDATE]
        
        # Return only "active" tracks (those updated this frame)
        # This is optional, but often `return self.tracks` is fine.
        active_tracks = [t for t in self.tracks if t.time_since_update == 0]
        return active_tracks