# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, os, copy, math
from intelligent_vehicles.trackers.ab3dmot.box import Box3D
from intelligent_vehicles.trackers.ab3dmot.matching import data_association
from intelligent_vehicles.trackers.ab3dmot.kalman_filter import KF

np.set_printoptions(suppress=True, precision=3)

# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):			  	
    def __init__(self):                    

        # counter
        self.trackers = []
        self.frame_count = 0
        self.ID_count = [0]
        self.id_now_output = []

        # config
        self.ego_com = False             # ego motion compensation
        self.affi_process = True        # post-processing affinity

        # debug
        # self.debug_id = 2
        self.debug_id = None
        self.algm = 'greedy' #hungar
        self.metric = 'giou_3d'

        # tracker parameters per category
        self.params = {
            'car':         {'thres': -0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': -1.0},
            'pedestrian':  {'thres': -0.5, 'min_hits': 1, 'max_age': 4, 'max_sim': 1.0, 'min_sim': -1.0},
            'cyclist':     {'thres':  -0.6, 'min_hits': 3, 'max_age': 4, 'max_sim': 0.0, 'min_sim': -100.},
            'bus':         {'thres': -0.3, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': -1.0},
            'truck':       {'thres': -0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': -1.0},
            'motorcycle':  {'thres': -0.7, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': -1.0},
            'default':     {'thres': -0.4, 'min_hits': 2, 'max_age': 3, 'max_sim': 1.0, 'min_sim': -1.0}
        }
        
    def reset(self):
        self.trackers = []
        self.frame_count = 0
        self.ID_count = [0]
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
    
    def within_range(self, theta):
		# make sure the orientation is within a proper range
        
        if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
        if theta < -np.pi: theta += np.pi * 2
        
        return theta
    
    def orientation_correction(self, theta_pre, theta_obs):
		# update orientation in propagated tracks and detected boxes so that they are within 90 degree
		
		# make the theta still in the range
        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)

		# if the angle of two theta is not acute angle, then make it acute
        if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:     
            theta_pre += np.pi       
            theta_pre = self.within_range(theta_pre)

		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0: theta_pre += np.pi * 2
            else: theta_pre -= np.pi * 2

        return theta_pre, theta_obs

    def ego_motion_compensation(self):
		# inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching
        pass



    def predict(self):
		# get predicted locations from existing tracks

        trks = []
        for t in range(len(self.trackers)):
			
			# propagate locations
            kf_tmp = self.trackers[t]
            kf_tmp.kf.predict()
            kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])

			# update statistics
            kf_tmp.time_since_update += 1 		
            trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks

    def update(self, matched, unmatched_trks, dets):
		# update matched trackers with assigned detections
		
        dets = copy.copy(dets)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
                assert len(d) == 1, 'error'

				# update statistics
                trk.time_since_update = 0		# reset because just updated
                trk.hits += 1

				# update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(dets[d[0]])
                trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])
                trk.kf.update(bbox3d)
                trk.kf.x[3] = self.within_range(trk.kf.x[3])
                #trk.confidence recompute confidence


    def intialize(self, dets, unmatched_dets):
		# create and initialise new trackers for unmatched detections

		# dets = copy.copy(dets)
        new_id_list = list()					# new ID generated for unmatched detections
        for i in unmatched_dets:        			# a scalar of index
            trk = KF(Box3D.bbox2array(dets[i]), self.ID_count[0])
            self.trackers.append(trk)
            new_id_list.append(trk.id)
			# print('track ID %s has been initialized due to new detection' % trk.id)

            self.ID_count[0] += 1

        return new_id_list

    def get_active_tracklets(self): #done
		# output exiting tracks that have been stably associated, i.e., >= min_hits
		# and also delete tracks that have appeared for a long time, i.e., >= max_age

        num_trks = len(self.trackers)
        results = []
        for trk in reversed(self.trackers):
			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
            d = Box3D.bbox2array_raw(d)

            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
                results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1)) 		
            num_trks -= 1

			# deadth, remove dead tracklet
            if (trk.time_since_update >= self.max_age): 
                self.trackers.pop(num_trks)

        return results

    def process_affi(self, affi, matched, unmatched_dets, new_id_list):

		# post-processing affinity matrix, convert from affinity between raw detection and past total tracklets
		# to affinity between past "active" tracklets and current active output tracklets, so that we can know 
		# how certain the results of matching is. The approach is to find the correspondes of ID for each row and
		# each column, map to the actual ID in the output trks, then purmute/expand the original affinity matrix
		
		###### determine the ID for each past track
        trk_id = self.id_past 			# ID in the trks for matching

		###### determine the ID for each current detection
        det_id = [-1 for _ in range(affi.shape[0])]		# initialization

		# assign ID to each detection if it is matched to a track
        for match_tmp in matched:		
            det_id[match_tmp[0]] = trk_id[match_tmp[1]]

		# assign the new birth ID to each unmatched detection
        count = 0
        assert len(unmatched_dets) == len(new_id_list), 'error'
        for unmatch_tmp in unmatched_dets:
            det_id[unmatch_tmp] = new_id_list[count] 	# new_id_list is in the same order as unmatched_dets
            count += 1
        assert not (-1 in det_id), 'error, still have invalid ID in the detection list'

		############################ update the affinity matrix based on the ID matching
		
		# transpose so that now row is past trks, col is current dets	
        affi = affi.transpose() 			

		###### compute the permutation for rows (past tracklets), possible to delete but not add new rows
        permute_row = list()
        for output_id_tmp in self.id_past_output:
            index = trk_id.index(output_id_tmp)
            permute_row.append(index)
        affi = affi[permute_row, :]	
        assert affi.shape[0] == len(self.id_past_output), 'error'

		###### compute the permutation for columns (current tracklets), possible to delete and add new rows
		# addition can be because some tracklets propagated from previous frames with no detection matched
		# so they are not contained in the original detection columns of affinity matrix, deletion can happen
		# because some detections are not matched

        max_index = affi.shape[1]
        permute_col = list()
        to_fill_col, to_fill_id = list(), list() 		# append new columns at the end, also remember the ID for the added ones
        for output_id_tmp in self.id_now_output:
            try:
                index = det_id.index(output_id_tmp)
            except:		# some output ID does not exist in the detections but rather predicted by KF
                index = max_index
                max_index += 1
                to_fill_col.append(index); to_fill_id.append(output_id_tmp)
            permute_col.append(index)

		# expand the affinity matrix with newly added columns
        append = np.zeros((affi.shape[0], max_index - affi.shape[1]))
        append.fill(self.min_sim)
        affi = np.concatenate([affi, append], axis=1)

		# find out the correct permutation for the newly added columns of ID
        for count in range(len(to_fill_col)):
            fill_col = to_fill_col[count]
            fill_id = to_fill_id[count]
            row_index = self.id_past_output.index(fill_id)

			# construct one hot vector because it is proapgated from previous tracks, so 100% matching
            affi[row_index, fill_col] = self.max_sim		
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

		# update trks with matched detection measurement
        self.update(matched, unmatched_trks, dets)

		# create and initialise new trackers for unmatched detections
        new_id_list = self.intialize(dets, unmatched_dets)

		# post-processing affinity to convert to the affinity between resulting tracklets
        if self.affi_process:
            affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)