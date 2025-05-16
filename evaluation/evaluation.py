#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 08:56:13 2024

@author: nadya
"""
import numpy as np
import os,sys
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligent_vehicles.trackers.ab3dmot.box import Box3D
from intelligent_vehicles.trackers.ab3dmot.dist_metrics import iou, dist3d, dist_ground, m_distance
from intelligent_vehicles.trackers.ab3dmot.matching import compute_affinity, data_association

def calculate_ade(predictions, ground_truth):
    """
    Calculate Average Displacement Error (ADE) for motion forecasting.
    
    Args:
    - predictions: List of predicted trajectories, each trajectory is a numpy array of shape (T, 2)
    - ground_truth: List of ground truth trajectories, each trajectory is a numpy array of shape (T, 2)
    
    Returns:
    - ade: Average Displacement Error
    """
    num_trajectories = len(ground_truth)
    total_pred = 0
    total_error = 0.0
    
    for i in range(num_trajectories):
        pred_traj = predictions[i]
        true_traj = ground_truth[i]
        
        if len(ground_truth[i]) > 0: 

            # Calculate Euclidean distance for each time step
            errors = np.linalg.norm(pred_traj - true_traj, axis=1)
            # Average displacement error for this trajectory
            total_error += np.sum(errors) / len(pred_traj)
            #print(pred_traj, true_traj, total_error)
            #print("###########################################################")
            total_pred += 1
    
    ade = total_error / total_pred

    return ade

def calculate_fde(predictions, ground_truth):
    """
    Calculate Final Displacement Error (FDE) for motion forecasting.
    
    Args:
    - predictions: List of predicted trajectories, each trajectory is a numpy array of shape (T, 2)
    - ground_truth: List of ground truth trajectories, each trajectory is a numpy array of shape (T, 2)
    
    Returns:
    - fde: Final Displacement Error
    """
    num_trajectories = len(predictions)
    total_error = 0.0
    total_pred = 0
    
    for i in range(num_trajectories):
        pred_traj = predictions[i][-1]  # Last position of predicted trajectory
        
        if len(ground_truth[i]) > 0: 
            
            true_traj = ground_truth[i][-1]  # Last position of ground truth trajectory
    
            # Calculate Euclidean distance between final positions
            error = np.linalg.norm(pred_traj - true_traj)
            total_error += error
            
            total_pred += 1
    
    fde = total_error / total_pred
    return fde





def read_results(results_folder, name):
    """
    Read ground truth and tracklet results from saved files
    
    Args:
        results_folder: Folder containing the result files
        name: Base name for the files
        
    Returns:
        gt_by_frame: Dictionary mapping frame_id to list of ground truth Box3D objects
        tracks_by_frame: Dictionary mapping frame_id to list of tracked Box3D objects
        gt_ids_by_frame: Dictionary mapping frame_id to list of ground truth IDs
        track_ids_by_frame: Dictionary mapping frame_id to list of track IDs
    """
    gt_file = os.path.join(results_folder, f"{name}_gt.txt")
    tracklets_file = os.path.join(results_folder, f"{name}_tracklets.txt")
    
    gt_by_frame = defaultdict(list)
    tracks_by_frame = defaultdict(list)
    gt_ids_by_frame = defaultdict(list)
    track_ids_by_frame = defaultdict(list)
    
    # Read ground truth
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                frame_id = int(data[0])
                obj_id = int(data[1])
                label = data[2]
                x, y, z = float(data[3]), float(data[4]), float(data[5])
                yaw = float(data[6])
                l, w, h = float(data[7]), float(data[8]), float(data[9])
                score = float(data[10])
                
                # Create Box3D object similar to how it's used in your tracker
                box = Box3D(h, w, l, x, y, z, yaw, score, label)
                box.obj_id = obj_id  # Add ID as attribute
                
                gt_by_frame[frame_id].append(box)
                gt_ids_by_frame[frame_id].append(obj_id)
    else:
        print(f"Warning: Ground truth file {gt_file} not found")
    
    # Read tracklets
    if os.path.exists(tracklets_file):
        with open(tracklets_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                frame_id = int(data[0])
                track_id = int(data[1])
                category = data[2]
                x, y, z = float(data[3]), float(data[4]), float(data[5])
                yaw = float(data[6])
                l, w, h = float(data[7]), float(data[8]), float(data[9])
                confidence = float(data[10])
                
                # Create Box3D object
                box = Box3D(h, w, l, x, y, z, yaw, confidence, category)
                box.obj_id = track_id  # Add track ID as attribute
                
                tracks_by_frame[frame_id].append(box)
                track_ids_by_frame[frame_id].append(track_id)
    else:
        print(f"Warning: Tracklets file {tracklets_file} not found")
    
    return gt_by_frame, tracks_by_frame, gt_ids_by_frame, track_ids_by_frame

def evaluate_mot(results_folder, name, metric='iou_3d', matching_threshold=0.25, visualization=True):
    """
    Evaluate Multiple Object Tracking metrics using saved results
    
    Args:
        results_folder: Folder containing the result files
        name: Base name for the files
        metric: Distance metric to use for matching ('iou_3d', 'dist_3d', etc.)
        matching_threshold: Threshold for considering a match valid
        visualization: Whether to visualize results
        
    Returns:
        Dictionary containing MOT metrics
    """
    # Read results
    gt_by_frame, tracks_by_frame, gt_ids_by_frame, track_ids_by_frame = read_results(results_folder, name)
    
    # Initialize metrics
    total_gt = 0
    total_matches = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_gt_paths = set()
    
    # For tracking precision
    total_distance_error = 0
    
    # For ID metrics
    last_matched_ids = {}  # Maps GT IDs to last matched track IDs
    gt_track_matches = defaultdict(list)  # Maps GT IDs to list of frames where they were matched
    gt_frames = defaultdict(list)  # Maps GT IDs to list of frames where they appear
    
    # Frame-by-frame metrics for visualization
    frame_metrics = {
        'frame_ids': [],
        'recall': [],
        'precision': [],
        'mota': [],
        'fp': [],
        'fn': [],
        'id_switches': []
    }

    params = {
            'car':         {'thres': 0.2, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'pedestrian':  {'thres': 0.2, 'min_hits': 1, 'max_age': 4, 'max_sim': 1.0, 'min_sim': 0.0},
            'cyclist':     {'thres': 0.2, 'min_hits': 3, 'max_age': 4, 'max_sim': 1.0, 'min_sim': 0.0},
            'bus':         {'thres': 0.2, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'van':         {'thres': 0.2, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'truck':       {'thres': 0.2, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'motorcycle':  {'thres': 0.2, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'default':     {'thres': 0.2, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0}
        }
    
    # Default class threshold for unknown classes
    default_class_thresh = {'thres': matching_threshold}
    
    # Process each frame
    all_frames = sorted(set(list(gt_by_frame.keys()) + list(tracks_by_frame.keys())))
    
    for frame_id in all_frames:
        gt_objects = gt_by_frame.get(frame_id, [])
        track_objects = tracks_by_frame.get(frame_id, [])
        
        # Count total GT objects
        total_gt += len(gt_objects)
        
        # Record GT objects in this frame for each ID
        for gt in gt_objects:
            gt_frames[gt.obj_id].append(frame_id)
            total_gt_paths.add(gt.obj_id)
        
        # Skip if no GT or tracks
        if not gt_objects or not track_objects:
            total_fn += len(gt_objects)
            total_fp += len(track_objects)
            
            # Add to frame metrics
            frame_metrics['frame_ids'].append(frame_id)
            frame_metrics['recall'].append(0 if len(gt_objects) > 0 else 1)
            frame_metrics['precision'].append(0 if len(track_objects) > 0 else 1)
            frame_metrics['mota'].append(1 - (len(gt_objects) + len(track_objects)) / max(1, len(gt_objects)))
            frame_metrics['fp'].append(len(track_objects))
            frame_metrics['fn'].append(len(gt_objects))
            frame_metrics['id_switches'].append(0)
            continue
        
        # Use the data_association function from matching.py
        for gt_obj in gt_objects:
            # Ensure all required parameters are present
            if not hasattr(gt_obj, 'obj_class') or gt_obj.obj_class is None:
                gt_obj.obj_class = gt_obj.type if hasattr(gt_obj, 'type') else 'car'  # Default to 'car' if not specified
                
            # Make sure params contains thresholds for this class
            if gt_obj.obj_class not in params:
                params[gt_obj.obj_class] = default_class_thresh
        
        # Make sure all track objects have a category attribute
        for track_obj in track_objects:
            if not hasattr(track_obj, 'category') or track_obj.category is None:
                track_obj.category = 'car'  # Default to 'car' if not specified
            track_obj.obj_class = track_obj.category  # Ensure obj_class is set for matching
        
        # Associate detections with tracks using the matching.py functions
        matches, unmatched_dets, unmatched_trks, _, _ = data_association(
            gt_objects, track_objects, params, 'greedy', metric, None, 1
        )

        print(f"Frame {frame_id}: Matches: {matches}, Unmatched Dets: {unmatched_dets}, Unmatched Trks: {unmatched_trks}")

        print(f"Frame {frame_id}: GTs: {len(gt_objects)}, Tracks: {len(track_objects)}")

        # print unmatched detections and unmatched tracks
        # for i in unmatched_dets:
        #     print(f"Unmatched Det: {gt_objects[i].obj_id} at frame {frame_id} is  {gt_objects[i].get_box3d()}")
        # for i in unmatched_trks:
        #     print(f"Unmatched Trk: {track_objects[i].id} at frame {frame_id} is  {track_objects[i].get_box3d()}")
        
        # Count matches, FP, FN
        total_matches += len(matches)
        total_fp += len(unmatched_trks)
        total_fn += len(unmatched_dets)
        
        # Check for ID switches
        frame_id_switches = 0
        for match in matches:
            gt_idx, trk_idx = match
            gt_id = gt_objects[gt_idx].obj_id
            track_id = track_objects[trk_idx].id
            
            # Record match for tracking stats
            gt_track_matches[gt_id].append(frame_id)
            
            # Check for ID switch
            if gt_id in last_matched_ids and last_matched_ids[gt_id] != track_id:
                total_id_switches += 1
                frame_id_switches += 1
            
            # Update last matched ID
            last_matched_ids[gt_id] = track_id
            
            # Calculate distance for MOTP (using 3D position)
            if metric == 'dist_3d':
                gt_pos = np.array([gt_objects[gt_idx].x, gt_objects[gt_idx].y, gt_objects[gt_idx].z])
                trk_pos = np.array([track_objects[trk_idx].x, track_objects[trk_idx].y, track_objects[trk_idx].z])
                dist = np.linalg.norm(gt_pos - trk_pos)
                total_distance_error += dist
            elif 'iou' in metric:
                # For IOU, we use 1 - IOU as the error
                iou_val = iou(gt_objects[gt_idx], track_objects[trk_idx], metric)
                total_distance_error += (1 - iou_val)
        
        # Add to frame metrics
        recall = len(matches) / max(1, len(gt_objects))
        precision = len(matches) / max(1, len(track_objects))
        mota = 1 - (len(unmatched_dets) + len(unmatched_trks) + frame_id_switches) / max(1, len(gt_objects))
        
        frame_metrics['frame_ids'].append(frame_id)
        frame_metrics['recall'].append(recall)
        frame_metrics['precision'].append(precision)
        frame_metrics['mota'].append(mota)
        frame_metrics['fp'].append(len(unmatched_trks))
        frame_metrics['fn'].append(len(unmatched_dets))
        frame_metrics['id_switches'].append(frame_id_switches)
    
    # Calculate overall metrics
    mota = 1 - (total_fn + total_fp + total_id_switches) / max(1, total_gt)
    motp = total_distance_error / max(1, total_matches)
    recall = total_matches / max(1, total_matches + total_fn)
    precision = total_matches / max(1, total_matches + total_fp)
    f1_score = 2 * precision * recall / max(0.001, precision + recall)
    
    # Calculate tracked stats
    mostly_tracked = 0
    partially_tracked = 0
    mostly_lost = 0
    
    for gt_id, frames in gt_frames.items():
        if gt_id in gt_track_matches:
            track_ratio = len(gt_track_matches[gt_id]) / len(frames)
            if track_ratio > 0.8:
                mostly_tracked += 1
            elif track_ratio >= 0.2:
                partially_tracked += 1
            else:
                mostly_lost += 1
        else:
            mostly_lost += 1
    
    # Prepare results
    results = {
        "MOTA": mota,
        "MOTP": motp,
        "Recall": recall,
        "Precision": precision,
        "F1": f1_score,
        "ID Switches": total_id_switches,
        "Mostly Tracked": mostly_tracked,
        "Partially Tracked": partially_tracked,
        "Mostly Lost": mostly_lost,
        "GT Objects": total_gt,
        "GT Tracks": len(total_gt_paths),
        "True Positives": total_matches,
        "False Positives": total_fp,
        "False Negatives": total_fn
    }
    
    # Display results
    print("\n--- MOT Evaluation Results ---")
    print(f"MOTA: {mota:.4f}")
    print(f"MOTP: {motp:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"ID Switches: {total_id_switches}")
    print(f"Mostly Tracked: {mostly_tracked}")
    print(f"Partially Tracked: {partially_tracked}")
    print(f"Mostly Lost: {mostly_lost}")
    
    # Optional visualization
    if visualization:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(frame_metrics['frame_ids'], frame_metrics['recall'], '-b', label='Recall')
        plt.plot(frame_metrics['frame_ids'], frame_metrics['precision'], '-r', label='Precision')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.title('Recall and Precision per Frame')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(frame_metrics['frame_ids'], frame_metrics['mota'], '-g', label='MOTA')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.title('MOTA per Frame')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(frame_metrics['frame_ids'], frame_metrics['fp'], '-r', label='False Positives')
        plt.plot(frame_metrics['frame_ids'], frame_metrics['fn'], '-b', label='False Negatives')
        plt.xlabel('Frame')
        plt.ylabel('Count')
        plt.title('FP and FN per Frame')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(frame_metrics['frame_ids'], frame_metrics['id_switches'], '-m', label='ID Switches')
        plt.xlabel('Frame')
        plt.ylabel('Count')
        plt.title('ID Switches per Frame')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{name}_mot_metrics.png'))
        plt.show()
    
    return results



from collections import defaultdict

from collections import defaultdict

def calculate_id_switches(gt_by_frame, trk_by_frame, params, metric='iou_3d'):
    id_switches = 0
    prev_trk_ids = {}  # gt_id -> trk_id
    trk_usage = {}     # trk_id -> last gt_id

    for frame_id in sorted(gt_by_frame.keys()):
        gt_objects = gt_by_frame[frame_id]
        track_objects = trk_by_frame.get(frame_id, [])
        print(f"Frame {frame_id}: GTs: {len(gt_objects)}, Tracks: {len(track_objects)}")
        
        matches, _, _, _, _ = data_association(
            gt_objects, track_objects, params, 'greedy', metric, None, 1
        )

        curr_matches = {}
        print(f"Frame {frame_id}: Matches: {matches}")


        for gt_idx, trk_idx in matches:
            gt = gt_objects[gt_idx]
            trk = track_objects[trk_idx]
            gt_id = gt.obj_id
            trk_id = trk.obj_id

            print(f"Frame {frame_id}: GT {gt_id} matched to Track {trk_id}")

            # Check if GT was previously matched to a different tracker ID → switch
            prev_trk_id = prev_trk_ids.get(gt_id)
            if prev_trk_id is not None and prev_trk_id != trk_id:
                id_switches += 1
                print(f"******************FID switch detected: GT {gt_id} matched to Track {trk_id} (previously matched to Track {prev_trk_id})**************************")

            # Check if the tracker was previously assigned to a different GT ID → switch
            last_gt_for_trk = trk_usage.get(trk_id)
            if last_gt_for_trk is not None and last_gt_for_trk != gt_id:
                id_switches += 1
                print(f"***********ID switch detected: GT {gt_id} matched to Track {trk_id} (previously matched to GT {last_gt_for_trk})************")

            curr_matches[gt_id] = trk_id
            # Update trk_usage to reflect the last GT ID for this tracker
            prev_trk_ids[gt_id] = trk_id
            trk_usage[trk_id] = gt_id

        # Update prev_trk_ids with current matches, but retain unmatched ones
        for gt_id in prev_trk_ids:
            if gt_id not in curr_matches:
                # Carry forward unmatched ones
                curr_matches[gt_id] = prev_trk_ids[gt_id]

        prev_trk_ids = curr_matches


    return id_switches


def main():
    # Tracker Evaluation
    results_folder = "results"
    name = "ego_vehicle"

    params = {
            'car':         {'thres': 0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'pedestrian':  {'thres': 0.5, 'min_hits': 1, 'max_age': 4, 'max_sim': 1.0, 'min_sim': 0.0},
            'cyclist':     {'thres': 0.6, 'min_hits': 3, 'max_age': 4, 'max_sim': 1.0, 'min_sim': 0.0},
            'bus':         {'thres': 0.3, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'van':         {'thres': 0.3, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'truck':       {'thres': 0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'motorcycle':  {'thres': 0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'default':     {'thres': 0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0}
        }
    # Evaluate MOT metrics
    # metrics = evaluate_mot(results_folder, name, visualization=True)

    #print("Metrics:", metrics)

    gt_by_frame, tracks_by_frame, gt_ids_by_frame, track_ids_by_frame = read_results(results_folder, name)
    # Calculate ID switches using data association
    id_switches = calculate_id_switches(
        gt_by_frame, tracks_by_frame, params, metric='iou_3d'
    )
    print(f"Total ID Switches: {id_switches}")
    
   
if __name__ == "__main__":
    main()