import numpy as np
from intelligent_vehicles.trackers.ab3dmot.box import Box3D
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from intelligent_vehicles.trackers.ab3dmot.dist_metrics import iou
import copy


class Track(object):
	def __init__(self, bbox3D, ID):
		self.time_since_update = 0
		self.id = ID
		self.hits = 1
		self.category = bbox3D.obj_class
		self.confidence = bbox3D.s
		self.initial_pos = Box3D.bbox2array(bbox3D)
		self.current_pos = Box3D.bbox2array(bbox3D)
		self.history = []
		self.max_length = 20
		self.w = [0.4,0.25,0.35] # weights for confidence update
		self.alpha = 0.4 # alpha for confidence update
		self.kalman_filter = KF(self.initial_pos)



	def get_3dbbox(self):
		# return the 3D bbox in the state
		tracked_bbox = Box3D()
		tracked_bbox.x, tracked_bbox.y, tracked_bbox.z, tracked_bbox.ry, tracked_bbox.l, tracked_bbox.w, tracked_bbox.h = self.current_pos #self.kalman_filter.kf.x[:7].reshape((7, ))
		tracked_bbox.s = self.confidence
		tracked_bbox.obj_class = self.category

		bbox = Box3D.bbox2array_raw(tracked_bbox) #[bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry]
		return bbox
    
	def track_bbox(self):
		# return the 3D bbox in the state
		tracked_bbox = Box3D()
		tracked_bbox.x, tracked_bbox.y, tracked_bbox.z, tracked_bbox.ry, tracked_bbox.l, tracked_bbox.w, tracked_bbox.h = self.current_pos
		#self.kalman_filter.kf.x[:7].reshape((7, ))
		tracked_bbox.s = self.confidence
		tracked_bbox.obj_class = self.category
		
		return tracked_bbox


	def update(self, detection,metric='iou'):

		self.time_since_update = 0
		self.hits += 1

		bbox3d = Box3D.bbox2array(detection)

		# update the current position of the track
		self.current_pos = copy.copy(bbox3d)

		# update orientation in propagated tracks and detected boxes so that they are within 90 degree
		self.kalman_filter.kf.x[3], bbox3d[3] = self.orientation_correction(self.kalman_filter.kf.x[3], bbox3d[3])
		# print(f"kalman filter x[3]: {self.kalman_filter.kf.x[:7]} bbox3d[3]: {bbox3d[3]}")
		
		self.kalman_filter.kf.update(bbox3d)
		# print(f"kalman filter x[3] after update: {self.kalman_filter.kf.x[:7]}")
		self.kalman_filter.kf.x[3] = self.within_range(self.kalman_filter.kf.x[3])
		# print(f"kalman filter x[3] after range: {self.kalman_filter.kf.x[:7]}")

		
		#update confidence
		self.update_confidence(detection,metric)

		# update the current position of the track which was predicted
		self.history[-1] = self.current_pos
		
		self.history = self.history[-self.max_length:]  # keep only the latest measurements


	def update_confidence(self, detection,metric='iou'):

		trk_det = Box3D.array2bbox(self.current_pos)
		
		
		calculated_iou = iou(trk_det, detection,metric)
		
		det_score = detection.s
		
		# Update confidence using determinant of covariance as uncertainty measure
		det_p = np.linalg.det(self.kalman_filter.kf.P)
		det_p = max(det_p, 1e-10)  # Avoid log(0)
		entropy_uncertainty = np.log(det_p)

		# Calculate trace-based uncertainty (sum of variances)
		trace_uncertainty = np.trace(self.kalman_filter.kf.P)
		old_confidence = self.confidence
		uncertanity_scale = (1/ (1.0 + trace_uncertainty))
		conf_new = self.w[0] *det_score + self.w[1] * calculated_iou + self.w[2]*uncertanity_scale
		self.confidence = self.alpha * self.confidence + (1 - self.alpha) * conf_new
		
		# print(f"\n\nUpdated tracker {self.id} with detection {detection}: \nuncertanity_scale: {uncertanity_scale:.3f}, old_confidence = {old_confidence:.3f}, trace_uncertainty = {trace_uncertainty:.3f}, IoU = {calculated_iou:.3f},new_confidence = {conf_new:.3f}, \nupdated confidence = {self.confidence:.3f}, det_score = {detection.s}")
	
	def predict(self):
		# print(f"predict track {self.id} with time since update {self.time_since_update} and current pos {self.kalman_filter.kf.x[:7].reshape((7, ))} ")
		# print("Before predict:")
		# print("State (x):", self.kalman_filter.kf.x)
		# print("Covariance (P):", np.diag(self.kalman_filter.kf.P))  # 
	
		self.kalman_filter.kf.predict()

		# print("After predict:")
		# print("State (x):", self.kalman_filter.kf.x)
		# print("Covariance (P):", np.diag(self.kalman_filter.kf.P))
		
		self.kalman_filter.kf.x[3] = self.within_range(self.kalman_filter.kf.x[3])

		# update statistics
		self.time_since_update += 1 		

		# update the current position of the track
		self.current_pos = copy.copy(self.kalman_filter.kf.x[:7].reshape((7, )))
		# print(f"after predict {self.kalman_filter.kf.x[:7].reshape((7, ))}")

		# add the new measurement to the history
		self.history.append(self.current_pos)
		self.history = self.history[-self.max_length:]  # keep only the lastest max length measurements
		
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
        
        
        
	def get_history(self,history_len):
		# get the history of the track
		# if history_len is None, return the whole history
		if history_len is None:
			history_len = self.max_length

		return_len = min(history_len, len(self.history))
		return self.history[-return_len:]
	

class KF(object):
	""" 
	Kalman filter for 3D tracking
	"""
	def __init__(self, initial_pos,dim_x=10, dim_z=7):

		self.kf = KalmanFilter(dim_x=10, dim_z=7)    
		 
	
		# state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
		# constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz 
		# while all others (theta, l, w, h, dx, dy, dz) remain the same
		self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix, dim_x * dim_x
		                      [0,1,0,0,0,0,0,0,1,0],
		                      [0,0,1,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])     

		# measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
		self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
		                      [0,1,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0]])

		# measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
		# self.kf.R[0:,0:] *= 10.   

		# initial state uncertainty at time 0
		# Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
		self.kf.P[7:, 7:] *= 1000. 	
		self.kf.P *= 10.

		# process uncertainty, make the constant velocity part more certain
		self.kf.Q[7:, 7:] *= 0.01

		# initialize data
		self.kf.x[:7] = initial_pos.reshape((7, 1))
		self.kf.x[7:] = np.random.normal(0, 0.01, (3, 1))

	def compute_innovation_matrix(self):
		""" compute the innovation matrix for association with mahalanobis distance
		"""
		return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

	def get_velocity(self):
		# return the object velocity in the state

		return self.kf.x[7:]
	