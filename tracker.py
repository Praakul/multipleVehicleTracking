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


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_track_id = 0

    def update(self, detections):
        for track in self.tracks:
            track.predict()

        matched_track_indices = set()
        matched_det_indices = set()

        if len(self.tracks) > 0 and len(detections) > 0:
            num_tracks = len(self.tracks)
            num_dets = len(detections)
            cost_matrix = np.zeros((num_tracks, num_dets))

            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    cost_matrix[i, j] = 1 - iou(track.bbox, det)

            # Gate the cost matrix: set entries above threshold to a large
            # value so the Hungarian algorithm will never assign them.
            GATE_VALUE = 1e5
            gated_cost = cost_matrix.copy()
            gated_cost[gated_cost > (1 - IOU_MATCH_THRESHOLD)] = GATE_VALUE

            row_ind, col_ind = linear_sum_assignment(gated_cost)

            for r, c in zip(row_ind, col_ind):
                # Only accept assignments that were not gated
                if gated_cost[r, c] < GATE_VALUE:
                    self.tracks[r].update(detections[c])
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)

        # Unmatched detections become new tracks
        unmatched_detections = [det for i, det in enumerate(detections) if i not in matched_det_indices]
        for det in unmatched_detections:
            new_track = Track(self.next_track_id, det)
            self.tracks.append(new_track)
            self.next_track_id += 1

        # Remove stale unmatched tracks that exceeded the age limit
        unmatched_track_indices = set(range(len(self.tracks))) - matched_track_indices
        self.tracks = [
            t for i, t in enumerate(self.tracks)
            if i not in unmatched_track_indices or t.time_since_update <= MAX_FRAMES_SINCE_UPDATE
        ]

        active_tracks = [t for t in self.tracks if t.time_since_update == 0]
        return active_tracks