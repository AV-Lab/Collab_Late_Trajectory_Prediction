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
				det_trk = Box3D.array2bbox(trk.current_pos)
				trk_class = trk.category # trk.obj_class

			# Check if classes match (if class attributes are found)
			det_class = det.obj_class # det.obj_class
               
			

			if det_class != trk_class:
					aff_matrix[d, t] = -float('inf')  # Very low affinity for mismatched classes
					# print(f"det {d}: {det}\n  trk {t} : {det_trk} aff_matrix { -float('inf') }\n\n")
					continue
			
            
			
			# choose to use different distance metrics
			# print(f"det {d}: {det}\n  trk {t} : {det_trk} \n\nclass: {det.obj_class} trk {trk_class}\n\n")
			if 'iou' in metric:    	  dist_now = iou(det, det_trk, metric)            
			elif metric == 'm_dis':   dist_now = -m_distance(det, det_trk, trk_inv_inn_matrices[t])
			elif metric == 'euler':   dist_now = -m_distance(det, det_trk, None)
			elif metric == 'dist_2d': dist_now = -dist_ground(det, det_trk)              	
			elif metric == 'dist_3d': dist_now = -dist3d(det, trk)              				
			else: assert False, 'error'
			aff_matrix[d, t] = dist_now

			# print(f"det {d}: {det}\n  trk {t} : {det_trk} aff_matrix {dist_now} class: {det.obj_class} trk {trk_class}\n\n")
			

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

def data_association(dets, trks, params, algm, metric, sec_association=False, verbose=False, trk_innovation_matrix=None, hypothesis=1, threshold=-1000):
    """
    Assigns detections to tracked objects.
    
    Parameters:
    -----------
    dets : list
        A list of Box3D objects representing detections.
    trks : list
        A list of Box3D objects representing tracks or just a list of tracks.
    params : dict
        Parameters dictionary containing thresholds for different object classes.
    algm : str
        Algorithm to use for assignment ('hungar' or 'greedy').
    metric : str
        Metric used for computing affinity between detections and tracks.
    sec_association : bool, optional
        Type of association. False for first-time association (high threshold),
        True for subsequent associations (low threshold). Default is 1False.
    trk_innovation_matrix : numpy.ndarray, optional
        Innovation matrix for tracks. Default is None.
    hypothesis : int, optional
        Hypothesis number. Default is 1.
    threshold : float, optional
        Fallback threshold value. Default is -1000.
        
    Returns:
    --------
    matches : numpy.ndarray
        Array of shape (N, 2) containing indices of matched detections and tracks.
    unmatched_dets : numpy.ndarray
        Array containing indices of unmatched detections.
    unmatched_trks : numpy.ndarray
        Array containing indices of unmatched tracks.
    cost : float
        Total assignment cost.
    aff_matrix : numpy.ndarray
        Affinity matrix used for assignment.
    """
    # Initialize affinity matrix with zeros
    aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    
    # Handle edge cases - empty detections or tracks
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), [], 0, aff_matrix
    if len(dets) == 0:
        return np.empty((0, 2), dtype=int), [], np.arange(len(trks)), 0, aff_matrix
    
    # Compute affinity matrix between detections and tracks
    aff_matrix = compute_affinity(dets, trks, metric)
    
    # Perform association based on the selected algorithm
    if algm == 'hungar':
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-aff_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)
    elif algm == 'greedy':
        # Greedy matching
        matched_indices = greedy_matching(-aff_matrix)
    else:
        assert False, f"Unknown algorithm: {algm}. Expected 'hungar' or 'greedy'."
    
    # Compute total assignment cost
    cost = 0
    for row_index in range(matched_indices.shape[0]):
        cost -= aff_matrix[matched_indices[row_index, 0], matched_indices[row_index, 1]]
    
    # Find unmatched detections and tracks
    unmatched_dets = [d for d, det in enumerate(dets) if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t, trk in enumerate(trks) if t not in matched_indices[:, 1]]
    


        
    if sec_association:
        print("--------------------------------------------")
    else:
         print("\n*********************************************")

    # Debug information about unmatched objects
    if verbose:
        
        print(f"----len(dets): {len(dets)} len(trks): {len(trks)} : {len(matched_indices)} matches\n"
              f"len unmatched_dets: {len(unmatched_dets)} len unmatched_trks: {len(unmatched_trks)}----")
        
        for i in range(len(unmatched_dets)):
            print(f"Unmatched detection in phase one maching {dets[unmatched_dets[i]]} with index {unmatched_dets[i]} \n")
        for i in range(len(unmatched_trks)):
            print(f"Unmatched track in phase one maching {trks[unmatched_trks[i]].current_pos} with index {unmatched_trks[i]}\n"
                f"history: {trks[unmatched_trks[i]].history}  \n")
    
    
	
	# Filter out matches with low affinity
    matches = []
    for m in matched_indices:
        det_class = dets[m[0]].obj_class
        # Use appropriate threshold based on association type
        thres = params[det_class]['low_thres'] if sec_association else params[det_class]['thres'] 
        if sec_association:
             print(f"det {dets[m[0]]} trk {trks[m[1]]} aff_matrix {aff_matrix[m[0], m[1]]} thres {thres} class {det_class}")
        
        if aff_matrix[m[0], m[1]] < thres:
            # Low affinity match - add to unmatched lists
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            # Valid match
            matches.append(m.reshape(1, 2))
            # Uncomment for debugging:
            # print(f"Tracks and Detections matched with cost {aff_matrix[m[0], m[1]]} < {thres}")
    
    # Format matches array
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    print(f"\nlen(dets): {len(dets)} len(trks): {len(trks)} : {len(matched_indices)} matches\n\n"
              f"len(unmatched_dets): {len(unmatched_dets)} len(unmatched_trks): {len(unmatched_trks)}")

    
    #  Debug information about unmatched objects: Check if all objects were matched and print appropriate message
    if len(unmatched_trks) == 0 and len(unmatched_dets) == 0:
        phase = "two" if sec_association else "one"
        print(f"\nAll objects matched successfully in phase {phase}!")

    if sec_association or (len(unmatched_trks) == 0 and len(unmatched_dets) == 0):
        print("*********************************************\n\n\n")
    return matches, np.array(unmatched_dets), np.array(unmatched_trks), cost, aff_matrix