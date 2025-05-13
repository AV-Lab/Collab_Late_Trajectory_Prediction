import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from intelligent_vehicles.trackers.ab3dmot.box import Box3D
from intelligent_vehicles.trackers.ab3dmot.dist_metrics import iou, dist3d, dist_ground, m_distance

def compute_affinity(dets, trks, metric, trk_inv_inn_matrices=None):
	# compute affinity matrix

	aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
	for d, det in enumerate(dets):
		for t, trk in enumerate(trks):

			# get bbox of tracklet from kalman filter
			if isinstance(trk, Box3D):
    			# trk is a Box3D object
				trk_class = trk.obj_class # trk.obj_class
				det_trk = trk
			else:
				kf_tmp = trk.kf	
				det_trk = Box3D.array2bbox(kf_tmp.x.reshape((-1))[:7])
				trk_class = trk.category # trk.obj_class

			# Check if classes match (if class attributes are found)
			det_class = det.obj_class # det.obj_class
			

			if det_class != trk_class:
					aff_matrix[d, t] = -float('inf')  # Very low affinity for mismatched classes
					continue
		
			# choose to use different distance metrics
			if 'iou' in metric:    	  dist_now = iou(det, det_trk, metric)            
			elif metric == 'm_dis':   dist_now = -m_distance(det, det_trk, trk_inv_inn_matrices[t])
			elif metric == 'euler':   dist_now = -m_distance(det, det_trk, None)
			elif metric == 'dist_2d': dist_now = -dist_ground(det, det_trk)              	
			elif metric == 'dist_3d': dist_now = -dist3d(det, trk)              				
			else: assert False, 'error'
			aff_matrix[d, t] = dist_now
			print(f"det_class {det_class} trk_class {trk_class} aff_matrix[{d}, {t}] = {aff_matrix[d, t] , iou(det, det_trk, metric) }")

	return aff_matrix

def greedy_matching(cost_matrix):
    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py

    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    # sort all costs and then convert to 2D
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    # assign matches one by one given the sorting, but first come first serves
    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        # if both id has not been matched yet
        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(matched_indices)

def data_association(dets, trks, params, algm, metric, trk_innovation_matrix=None, hypothesis=1,threshold = -1000):   
	"""
	Assigns detections to tracked object

	dets:  a list of Box3D object
	trks:  a list of Box3D object

	Returns 3 lists of matches, unmatched_dets and unmatched_trks, and total cost, and affinity matrix
	"""

	# if there is no item in either row/col, skip the association and return all as unmatched
	aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
	if len(trks) == 0: 
		return np.empty((0, 2), dtype=int), np.arange(len(dets)), [], 0, aff_matrix
	if len(dets) == 0: 
		return np.empty((0, 2), dtype=int), [], np.arange(len(trks)), 0, aff_matrix		

	# compute affinity matrix
	aff_matrix = compute_affinity(dets, trks, metric)
	print(f"aff_matrix {aff_matrix}")
	

	# association based on the affinity matrix
	if algm == 'hungar':
		row_ind, col_ind = linear_sum_assignment(-aff_matrix)      	# hougarian algorithm
		matched_indices = np.stack((row_ind, col_ind), axis=1)
	elif algm == 'greedy':
		matched_indices = greedy_matching(-aff_matrix) 				# greedy matching
	else: assert False, 'error'

	# compute total cost
	cost = 0
	for row_index in range(matched_indices.shape[0]):
		cost -= aff_matrix[matched_indices[row_index, 0], matched_indices[row_index, 1]]

	# save for unmatched objects
	unmatched_dets = []
	for d, det in enumerate(dets):
		if (d not in matched_indices[:, 0]): unmatched_dets.append(d)
	unmatched_trks = []
	for t, trk in enumerate(trks):
		if (t not in matched_indices[:, 1]): unmatched_trks.append(t)

	# filter out matches with low affinity
	matches = []
	# print(f"matched_indices {matched_indices}")
	for m in matched_indices:
		det_class = dets[m[0]].obj_class
		thres = params[det_class]['thres']
		if (aff_matrix[m[0], m[1]] < thres):
			unmatched_dets.append(m[0])
			unmatched_trks.append(m[1])
		else: matches.append(m.reshape(1, 2))
	if len(matches) == 0: 
		matches = np.empty((0, 2),dtype=int)
	else: matches = np.concatenate(matches, axis=0)

	# print(f"matches {matches}")

	return matches, np.array(unmatched_dets), np.array(unmatched_trks), cost, aff_matrix