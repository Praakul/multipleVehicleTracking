import numpy as np

def iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Bboxes are expected in [x1, y1, x2, y2] format.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area

class KalmanFilter:
    def __init__(self):
        # State vector: [x_center, y_center, scale, ratio, vx, vy, vs]
        # We track center, scale (area), aspect ratio, and velocities of x, y, and scale.
        # This is a 7-dimensional state vector.
        self.state = np.zeros(7)
        
        # State transition matrix (models constant velocity)
        # [ 1, 0, 0, 0, dt, 0,  0 ]  (x' = x + vx*dt)
        # [ 0, 1, 0, 0, 0,  dt, 0 ]  (y' = y + vy*dt)
        # [ 0, 0, 1, 0, 0,  0,  dt]  (s' = s + vs*dt)
        # [ 0, 0, 0, 1, 0,  0,  0 ]  (r' = r)
        # [ 0, 0, 0, 0, 1,  0,  0 ]  (vx' = vx)
        # [ 0, 0, 0, 0, 0,  1,  0 ]  (vy' = vy)
        # [ 0, 0, 0, 0, 0,  0,  1 ]  (vs' = vs)
        # We assume dt=1 frame.
        self.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 0, 1],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1]])

        # Measurement matrix (maps state to measurement space)
        # We only measure [x_center, y_center, scale, ratio].
        # This is a 4-dimensional measurement.
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])

        # Initial state covariance (our uncertainty about the initial state)
        # Small uncertainty for observed variables (position, size), massive for unobserved (velocities)
        self.P = np.diag([10, 10, 10, 10, 10000, 10000, 10000]).astype(float)

        # Process noise covariance (uncertainty in the model)
        # This accounts for accelerations (changes in velocity).
        # Standard tuned hyperparameters for SORT process noise
        self.Q = np.diag([0.01, 0.01, 0.0001, 0.0001, 0.01, 0.01, 0.0001]).astype(float)

        # Measurement noise covariance (uncertainty from the YOLO detector)
        # This is how much we trust our detector's bounding boxes.
        r_val = 5.0 # Tune this based on detector accuracy
        self.R = np.eye(4) * r_val

    def predict(self):
        # Predict the next state
        self.state = self.F @ self.state
        # Predict the next state covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement):
        # --- Kalman filter update step ---
        
        # 1. Measurement residual (the error)
        y = measurement - (self.H @ self.state)
        
        # 2. Residual covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # 3. Kalman Gain (how much to trust the measurement)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 4. Update state estimate
        self.state = self.state + K @ y
        
        # 5. Update state covariance
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P