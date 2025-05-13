import numpy as np
from numba import jit
from copy import deepcopy

class Box3D:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, s=None,obj_class=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.ry = ry    # orientation
        self.s = s      # detection score
        self.obj_class = None  # object class 
        self.corners_3d_cam = None

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s)
    
    def get_box3d(self):
        return np.array([self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s,self.obj_class])
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.ry}
    
    @classmethod
    def bbox2array(cls, bbox):
        # if bbox.s is None:
        return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        # else:
        #     return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])

    @classmethod
    def bbox2array_raw(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])
        else:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s,bbox.obj_class])

    @classmethod
    def array2bbox_raw(cls, data):
        # take the format of data of [h,w,l,x,y,z,theta]

        bbox = Box3D()
        bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[7]

        if len(data) == 9:
            bbox.s = data[7]
            bbox.obj_class = data[-1]
        return bbox
    
    @classmethod
    def array2bbox(cls, data):
        # take the format of data of [x,y,z,theta,l,w,h]

        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        
        if len(data) == 9:
            bbox.s = data[7]
            bbox.obj_class = data[-1]
            
        return bbox
    

    @staticmethod
    def roty(t):
        """
        Create a 3x3 rotation matrix for rotation about the Y-axis (yaw)
        
        Args:
            t: rotation angle in radians
    
        Returns:
            3x3 numpy array
        """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([
            [ c, 0,  s],
            [ 0, 1,  0],
            [-s, 0,  c]
        ])
        
    @classmethod
    def box2corners3d_camcoord(cls, bbox):
        ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and 
            convert it to the 8 corners of the 3D box, the box is in the camera coordinate
            with right x, down y, front z
            
            Returns:
                corners_3d: (8,3) array in in rect camera coord

            box corner order is like follows
                    1 -------- 0         top is bottom because y direction is negative
                   /|         /|
                  2 -------- 3 .
                  | |        | |
                  . 5 -------- 4
                  |/         |/
                  6 -------- 7    
            
            rect/ref camera coord:
            right x, down y, front z

            x -> w, z -> l, y -> h
        '''

        # print(f"box2corners3d_camcoord cls :{cls}\n bbox: {bbox}")
        # print(f"box2corners3d_camcoord bbox.ry: {bbox.ry}")
        # if already computed before, then skip it
        if bbox.corners_3d_cam is not None:
            return bbox.corners_3d_cam

        # compute rotational matrix around yaw axis
        # -1.57 means straight, so there is a rotation here
        R = cls.roty(bbox.ry)   

        # 3d bounding box dimensions
        l, w, h = bbox.l, bbox.w, bbox.h

        # 3d bounding box corners
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
        y_corners = [0,0,0,0,-h,-h,-h,-h];
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] = corners_3d[0,:] + bbox.x
        corners_3d[1,:] = corners_3d[1,:] + bbox.y
        corners_3d[2,:] = corners_3d[2,:] + bbox.z
        corners_3d = np.transpose(corners_3d)
        bbox.corners_3d_cam = corners_3d

        return corners_3d
    
import numpy as np

# Create a simplified MOT evaluation function
def evaluate_mot(gt_data, pred_data, iou_threshold=0.5):
    """
    Evaluate MOT (Multiple Object Tracking) metrics
    
    Args:
        gt_data: List of frames, each frame is a list of ground truth boxes
        pred_data: List of frames, each frame is a list of predicted boxes
        iou_threshold: IoU threshold for considering a match
        
    Returns:
        Dictionary of MOT metrics
    """
    # Counters for metrics
    total_gt = 0
    total_pred = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    
    # Previous frame matches (track ID -> detection ID mapping)
    prev_matches = {}
    
    # Go through each frame
    for frame_idx, (gt_frame, pred_frame) in enumerate(zip(gt_data, pred_data)):
        # Count total objects
        total_gt += len(gt_frame)
        total_pred += len(pred_frame)
        
        # In a real implementation, we'd calculate 3D IoU here
        # For simplicity, we'll use dummy IoU values
        n_gt = len(gt_frame)
        n_pred = len(pred_frame)
        
        # Create a random IoU matrix for demonstration
        np.random.seed(frame_idx)  # For reproducibility
        iou_matrix = np.random.random((n_gt, n_pred)) * 0.8  # Random IoUs between 0 and 0.8
        
        # Apply greedy matching
        matched_gt = set()
        matched_pred = set()
        matches = []  # List of (gt_idx, pred_idx) matches
        
        # Match detections to ground truth
        while True:
            if len(matched_gt) == n_gt or len(matched_pred) == n_pred:
                break
                
            # Make a copy of IoU matrix and mask already matched pairs
            remaining_iou = iou_matrix.copy()
            for i in matched_gt:
                remaining_iou[i, :] = -1
            for j in matched_pred:
                remaining_iou[:, j] = -1
                
            # If no matches above threshold remain, break
            if np.max(remaining_iou) < iou_threshold:
                break
                
            # Get indices of max IoU
            i, j = np.unravel_index(np.argmax(remaining_iou), remaining_iou.shape)
            
            matched_gt.add(i)
            matched_pred.add(j)
            matches.append((i, j))
            true_positives += 1
        
        # Count false positives and negatives
        frame_false_positives = n_pred - len(matched_pred)
        frame_false_negatives = n_gt - len(matched_gt)
        
        false_positives += frame_false_positives
        false_negatives += frame_false_negatives
        
        # Check for ID switches (simplified)
        # In real implementation, we'd track object IDs across frames
        current_matches = {}
        for i, j in matches:
            gt_id = i  # Using index as ID for this simplified example
            pred_id = j
            current_matches[gt_id] = pred_id
            
            # Check if this gt_id had a different match in previous frame
            if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                id_switches += 1
        
        prev_matches = current_matches
    
    # Calculate overall metrics
    mota = 1 - (false_positives + false_negatives + id_switches) / max(1, total_gt)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    
    return {
        "MOTA": mota,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ID Switches": id_switches,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "True Positives": true_positives,
        "Total GT": total_gt,
        "Total Pred": total_pred
    }

# Create sample data
# In real implementation, these would be 3D bounding boxes
gt_data = [
    [1, 2],        # Frame 1: 2 objects 
    [1, 2],        # Frame 2: 2 objects
    [1, 2, 3],     # Frame 3: 3 objects
    [1, 3]         # Frame 4: 2 objects
]

pred_data = [
    [1, 2],        # Frame 1: 2 objects (correct)
    [1, 3],        # Frame 2: 2 objects (1 false positive, 1 correct)
    [1, 2],        # Frame 3: 2 objects (1 false negative)
    [1, 2, 4]      # Frame 4: 3 objects (1 false positive)
]

# Calculate metrics
metrics = evaluate_mot(gt_data, pred_data)

# Print results
print("MOT Evaluation Results:")
print("-" * 40)
for metric, value in metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

# Let's create another example with perfect tracking
print("\nPerfect tracking example:")
perfect_pred = gt_data.copy()
perfect_metrics = evaluate_mot(gt_data, perfect_pred)
print("-" * 40)
for metric, value in perfect_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

# Example with really poor tracking
print("\nPoor tracking example:")
poor_pred = [
    [5, 6],        # Frame 1: 2 completely wrong objects
    [7, 8],        # Frame 2: 2 completely wrong objects
    [9],           # Frame 3: 1 completely wrong object (also a false negative)
    [10, 11, 12]   # Frame 4: 3 completely wrong objects (1 false positive)
]
poor_metrics = evaluate_mot(gt_data, poor_pred)
print("-" * 40)
for metric, value in poor_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")