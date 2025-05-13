import numpy as np
from intelligent_vehicles.trackers.ab3dmot.box import Box3D
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from intelligent_vehicles.trackers.ab3dmot.dist_metrics import iou



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
		self.alpha = 0.5
		self.kalman_filter = KF(self.initial_pos)



	def get_3dbbox(self):
		# return the 3D bbox in the state
		tracked_bbox = Box3D()
		tracked_bbox.x, tracked_bbox.y, tracked_bbox.z, tracked_bbox.ry, tracked_bbox.l, tracked_bbox.w, tracked_bbox.h = self.kalman_filter.kf.x[:7].reshape((7, ))
		tracked_bbox.s = self.confidence
		tracked_bbox.obj_class = self.category

		bbox = Box3D.bbox2array_raw(tracked_bbox) #[bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry]
		return bbox


	def update(self, detection,metric='iou'):
		# update the kalman filter with the new measurement
		# self.kalman_filter.kf.predict()
		# self.kalman_filter.kf.update(bbox3D)

		self.time_since_update = 0
		self.hits += 1

		# update orientation in propagated tracks and detected boxes so that they are within 90 degree
		bbox3d = Box3D.bbox2array(detection)
		self.kf.x[3], bbox3d[3] = self.orientation_correction(self.kf.x[3], bbox3d[3])
		self.kf.update(bbox3d)
		self.kf.x[3] = self.within_range(self.kf.x[3])

		
		#update confidence
		self.update_confidence(detection,metric)
		
		# update the current position of the track
		self.current_pos = self.kf.x[:7].reshape((7, ))
	

		# update the current position of the track which was predicted
		self.history[-1] = self.current_pos
		
		self.history = self.history[-self.max_length:]  # keep only the latest measurements


	def update_confidence(self, detection,metric='iou'):

		trk_det = Box3D.array2bbox(self.current_pos)
		
		calculated_iou = iou(trk_det, detection,metric)
		
		det_score = detection.s
		
		# Update confidence using determinant of covariance as uncertainty measure
		det_p = np.linalg.det(self.kf.P)
		entropy_uncertainty = np.log(det_p)

		# Calculate trace-based uncertainty (sum of variances)
		trace_uncertainty = np.trace(self.kf.P)

		conf_new = det_score * calculated_iou / (1.0 + entropy_uncertainty)
		self.confidence = self.alpha * self.confidence + (1 - self.alpha) * conf_new

		print(f"Updated tracker {self.id} with detection {detection}: confidence = {self.confidence:.3f}, IoU = {calculated_iou:.3f}, det_p = {det_p:.3f} det_score = {detection.s}")
	
	def predict(self):
		self.kalman_filter.kf.predict()
		
		self.kalman_filter.kf.x[3] = self.within_range(self.kalman_filter.kf.x[3])

		# update statistics
		self.time_since_update += 1 		

		# update the current position of the track
		self.current_pos = self.kalman_filter.kf.x[:7].reshape((7, ))

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

	def ego_motion_compensation(self):
			# inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching
			pass
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

		self.kf = KalmanFilter(dim_x=10, dim_z=7,)       
	
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

	def compute_innovation_matrix(self):
		""" compute the innovation matrix for association with mahalanobis distance
		"""
		return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

	def get_velocity(self):
		# return the object velocity in the state

		return self.kf.x[7:]
	