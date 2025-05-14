# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, os, copy, math
from intelligent_vehicles.trackers.ab3dmot.box import Box3D
from intelligent_vehicles.trackers.ab3dmot.matching import data_association
from intelligent_vehicles.trackers.ab3dmot.Tracker import Track
from intelligent_vehicles.trackers.ab3dmot.dist_metrics import iou
from logging_setup import setup_logging

np.set_printoptions(suppress=True, precision=3)

# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):			  	
    def __init__(self):                    
   
        self.frame_count = 0
        self.ID_count = 0
        self.id_now_output = []

        # config
        self.ego_com = False             # ego motion compensation
        self.affi_process = True        # post-processing affinity

        # debug
        # self.debug_id = 2
        self.debug_id = None
        self.algm = 'greedy' #hungar
        self.metric = 'iou_3d'

        # tracker parameters per category
        self.params = {
            'car':         {'thres': 0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'pedestrian':  {'thres': 0.5, 'min_hits': 1, 'max_age': 4, 'max_sim': 1.0, 'min_sim': 0.0},
            'cyclist':     {'thres': 0.6, 'min_hits': 3, 'max_age': 4, 'max_sim': 1.0, 'min_sim': 0.0},
            'bus':         {'thres': 0.3, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'van':         {'thres': 0.3, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'truck':       {'thres': 0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'motorcycle':  {'thres': 0.7, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0},
            'default':     {'thres': 0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': 0.0}
        }
        
    def reset(self):
        self.trackers = []
        self.frame_count = 0
        self.ID_count = 0
        self.id_now_output = [] 
        
    def process_detections(self, dets):
		# convert each detection into the class Box3D 
		# inputs: 
		# 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]  
        dets_new = []
        for det in dets:
            det_tmp = Box3D.array2bbox_raw(det)
            dets_new.append(det_tmp)
            
        return dets_new
    


    def predict(self):
		# get predicted locations from existing tracks

        trks = []
        for t in range(len(self.trackers)):
			
			# propagate locations
            track = self.trackers[t]
            track.predict()
            trks.append(track)

        return trks

    def update(self, matched, unmatched_trks, dets):
		# update matched trackers with assigned detections
        dets = copy.copy(dets)
        # f"matched {matched} unmatched_trks {unmatched_trks}"
        # for idx in unmatched_trks:
        #     # print(f"------ALL unmatched track {self.trackers[idx].id} {self.trackers[idx].category} {self.trackers[idx].time_since_update} {self.trackers[idx].hits}")
        print(f"------------------------self.trackers {len(self.trackers)} dets: {len(dets)}---------------------------------")
        for t, trk in enumerate(self.trackers):
            
            if t not in unmatched_trks:
                
                d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
                assert len(d) == 1, 'error'

                detection = dets[d[0]]

                #update tracker with the new detection
                trk.update(detection,self.metric)
            
            else:
                # print(f"This track is not matched track!")
                # print(f"trk {t} {trk.id} {trk.category} {trk.time_since_update} {trk.hits}")
                # # check track age has reached max_age
                if trk.time_since_update >= self.params[trk.category]['max_age']:

                    # self.id_now_output.remove(trk.id)
                    # print("id_now_output",self.id_now_output)
                    # print(f"remove track {trk.id} {trk.category} {trk.time_since_update}")
                    # remove the track
                    self.trackers.pop(t)
                    # self.ID_count -= 1
        print(f"------------------Final shape of self.trackers {len(self.trackers)}")
                 


                
    def intialize(self, dets, unmatched_dets):
		# create and initialise new trackers for unmatched detections

		# dets = copy.copy(dets)
      
        new_id_list = list()		# new ID generated for unmatched detections
        for i in unmatched_dets: 
            # a scalar of index
            label = dets[i].obj_class
            score = dets[i].s
            trk = Track(dets[i],self.ID_count)
            self.trackers.append(trk)
            new_id_list.append(trk.id)
	

            self.ID_count += 1

        return new_id_list

    def get_active_tracklets(self, return_trks=True):
        """
        Gets active tracklets based on tracking parameters.
        
        This function:
        1. Returns existing tracks that have been stably associated (hits >= min_hits)
        2. Removes tracks that have not been updated for too long (time_since_update >= max_age)
        
        Args:
            return_trks (bool): If True, returns tracklet objects; otherwise returns 3D bounding boxes
            
        Returns:
            Results can be either:
            - List of tracklet objects (if return_trks=True)
            - Numpy array of 3D bounding boxes (if return_trks=False)
        """
        num_trks = len(self.trackers)
        results = []
        
        for trk in reversed(self.trackers):
            # Convert from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            d = Box3D.array2bbox(trk.kalman_filter.kf.x[:7].reshape((7, )))     # bbox location from Kalman filter
            d = Box3D.bbox2array_raw(d)
            obj_class = trk.category

            # Check if tracklet is active (recently updated and has enough hits)
            if ((trk.time_since_update < self.params[obj_class]['max_age']) and 
                (trk.hits >= self.params[obj_class]['min_hits'] or self.frame_count <= self.params[obj_class]['min_hits'])):      
                
                # Append either the tracklet or its 3D bbox based on return_trks
                results.append(trk) if return_trks else results.append(trk.get_3dbbox())  # Returns 3dbbox with confidence and class
                        
            num_trks -= 1

            # Remove dead tracklet that hasn't been updated for too long
            if (trk.time_since_update >= self.params[obj_class]['max_age']): 
                self.trackers.pop(num_trks)
        
        # Convert to numpy array if returning bounding boxes, otherwise keep as list of tracklets
        results = np.array(results) if not return_trks else results

        return results

    def process_affi(self, affi, matched, unmatched_dets, new_id_list, dets):
        """
        Post-processes the affinity matrix to convert from raw detection-to-track affinities
        to affinities between active past tracklets and current active output tracklets.
        
        This allows us to determine the certainty of matching results. The approach maps IDs
        for each row and column to the actual IDs in the output tracks, then permutes and 
        expands the original affinity matrix.
        
        Args:
            affi: Original affinity matrix (detections x tracks)
            matched: Array of matched detection and track indices
            unmatched_dets: Array of unmatched detection indices
            new_id_list: List of new IDs to assign to unmatched detections
            dets: List of detection objects with class information
            
        Returns:
            Processed affinity matrix
        """

        logger = setup_logging("collaboration.log")
        # Determine IDs for past tracks
        trk_id = self.id_past  # IDs in the tracks for matching

        # Determine IDs for current detections (initialize with -1)
        det_id = [-1 for _ in range(affi.shape[0])]

        # Assign IDs to matched detections
        for match_tmp in matched:
            det_id[match_tmp[0]] = trk_id[match_tmp[1]]

        # Assign new IDs to unmatched detections
        assert len(unmatched_dets) == len(new_id_list), 'Error: unmatched_dets and new_id_list must have same length'
        for i, unmatch_tmp in enumerate(unmatched_dets):
            det_id[unmatch_tmp] = new_id_list[i]  # new_id_list is in the same order as unmatched_dets
        
        assert not (-1 in det_id), 'Error: still have invalid ID in the detection list'

        # Update the affinity matrix based on ID matching
        
        # Transpose so rows represent past tracks, columns represent current detections
        affi = affi.transpose()

        # Compute permutation for rows (past tracklets) - possible to delete but not add new rows
        permute_row = []
        for output_id_tmp in self.id_past_output:
            index = trk_id.index(output_id_tmp)
            permute_row.append(index)
        
        affi = affi[permute_row, :]
        assert affi.shape[0] == len(self.id_past_output), 'Error: row count mismatch after permutation'

        # Compute permutation for columns (current tracklets) - possible to delete and add new columns
        # Additions occur when tracks propagated from previous frames have no matched detection
        max_index = affi.shape[1]
        permute_col = []
        to_fill_col, to_fill_id = [], []  # Track new columns and their IDs
        
        for output_id_tmp in self.id_now_output:
            try:
                index = det_id.index(output_id_tmp)
            except ValueError:  # ID doesn't exist in detections (predicted by Kalman filter)
                index = max_index
                max_index += 1
                to_fill_col.append(index)
                to_fill_id.append(output_id_tmp)
            permute_col.append(index)

        # Create mapping from detection ID to object class
        id_to_class = {}
        for idx, d_id in enumerate(det_id):
            if idx < len(dets):
                id_to_class[d_id] = dets[idx].obj_class

        # Expand the affinity matrix with newly added columns
        appended = np.zeros((affi.shape[0], max_index - affi.shape[1]))
        
        # Fill each row with class-specific minimum similarity values
        for i in range(appended.shape[0]):
            track_id = self.id_past_output[i]
            # Get the class for this track (with fallback to default if not found)
            if track_id in id_to_class:
                track_class = id_to_class[track_id]
            else:
                # Use a default class if this track ID isn't in our mapping
                track_class = next(iter(self.params.keys()))  # First available class as default
            
            # Fill this row with the appropriate min_sim
            appended[i, :].fill(self.params[track_class]['min_sim'])
        
        # Concatenate with original affinity matrix
        affi = np.concatenate([affi, appended], axis=1)

        # Set high similarity for propagated tracks
        for count in range(len(to_fill_col)):
            fill_col = to_fill_col[count]
            fill_id = to_fill_id[count]
            row_index = self.id_past_output.index(fill_id)
            
            # Get the class for this track (with fallback)
            if fill_id in id_to_class:
                track_class = id_to_class[fill_id]
                # Use class-specific max_sim
                affi[row_index, fill_col] = self.params[track_class]['max_sim']
            else:
                # Use default max_sim if class not found
                logger.warning(f"Track ID {fill_id} not found in id_to_class mapping. Using default max_sim.")
                affi[row_index, fill_col] = self.params['car']['max_sim']

        # Apply final column permutation
        affi = affi[:, permute_col]

        return affi
    def track(self, detections, ego_pose):
        """
        Params:
            dets_all: dict
				dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
				info: a array of other info for each det
			frame:    str, frame number, used to query ego pose
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""	

        self.frame_count += 1

		# recall the last frames of outputs for computing ID correspondences during affinity processing
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]

		# process detection format
        dets = self.process_detections(detections)

		# tracks propagation based on velocity
        trks = self.predict()

		# ego motion compensation
        if self.ego_com and ego_pose is not None:
            trks = self.ego_motion_compensation(trks)

		# matching
        matched, unmatched_dets, unmatched_trks, cost, affi = data_association(dets, trks, self.params, self.algm, self.metric)
        print(f"trks {len(trks)} dets: {len(dets)} matched {len(matched)}  unmatched_dets: {len(unmatched_dets)} unmatched_trks : {len(unmatched_trks)} {cost} {affi.shape} {self.frame_count}")

		# update trks with matched detection measurement
        self.update(matched, unmatched_trks, dets)

		# create and initialise new trackers for unmatched detections
        new_id_list = self.intialize(dets, unmatched_dets)

        print(f"-------number of tracks in trackers {len(self.trackers)}")
        print(f"-------number of active tracks {len(self.get_active_tracklets(False))} number of new tracks: {len(unmatched_dets)}")
        

		# post-processing affinity to convert to the affinity between resulting tracklets
        if self.affi_process:
            affi = self.process_affi(affi, matched, unmatched_dets, new_id_list, dets)