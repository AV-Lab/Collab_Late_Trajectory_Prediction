import numpy as np
from intelligent_vehicles.trackers.ab3dmot.box import Box3D
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints


# confidence estimation algorithm 

class Filter(object):
	def __init__(self, bbox3D, label, ID, confidence):
		self.time_since_update = 0
		self.id = ID
		self.hits = 1
		self.category = label
		self.confidence = confidence
		self.initial_pos = bbox3D
        

class KF(Filter):
	def __init__(self, bbox3D,  ID,score,label):
		super().__init__(bbox3D, label, ID,score)

		self.kf = KalmanFilter(dim_x=10, dim_z=7)       
		# There is no need to use EKF here as the measurement and state are in the same space with linear relationship

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
		self.kf.x[:7] = self.initial_pos.reshape((7, 1))

	def compute_innovation_matrix(self):
		""" compute the innovation matrix for association with mahalanobis distance
		"""
		return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

	def get_velocity(self):
		# return the object velocity in the state

		return self.kf.x[7:]
	
	def get_3dbbox(self):
		# return the 3D bbox in the state
		tracked_bbox = Box3D()
		tracked_bbox.x, tracked_bbox.y, tracked_bbox.z, tracked_bbox.ry, tracked_bbox.l, tracked_bbox.w, tracked_bbox.h = self.kf.x[:7].reshape((7, ))
		tracked_bbox.s = self.confidence
		tracked_bbox.obj_class = self.category

		bbox = Box3D.bbox2array_raw(tracked_bbox) #[bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry]
		
	
		return bbox
