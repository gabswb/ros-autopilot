import sys
import json
import yaml
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

import rospy
import tf2_ros
import transforms3d.quaternions as quaternions

class lane_side(str, Enum):
	FRONT = "front"
	LEFT = "left"
	RIGHT = "right"
	AWAY = "away"

class MapHandler(object):
	def __init__(self, config):
		self.config = config
		with open(self.config["map"]["road-network-path"], 'r') as f:
				self.road_network = json.load(f)
		self.lane_width = self.config["map"]["lane-width"]
		
		self.left_boundaries_x = []
		self.left_boundaries_y = []
		self.right_boundaries_x = []
		self.right_boundaries_y = []

		self.process_road_network()
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		# self.prev_x = 0
		# self.prev_y = 0
		# self.fig, self.ax = plt.subplots()
		# self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

		# self.plot_road()
		
	def get_lane_side(self, object):
		if np.abs(object.x) <= self.lane_width/2 and object.z >= 0:
			return lane_side.FRONT
		elif object.x < -self.lane_width/2 and object.x >= -self.lane_width*3/2:
			return lane_side.LEFT
		elif object.x > self.lane_width/2 and object.x <= self.lane_width*3/2:
			return lane_side.RIGHT
		else:
			return lane_side.AWAY
		
	def is_on_road(self, target_position):
		'''Return true if target_position is on the road
			- target_position: 2d-tuple (x,y)
		'''
		for left_boundary_x, left_boundary_y, right_boundary_x, right_boundary_y in zip(self.left_boundaries_x, self.left_boundaries_y, self.right_boundaries_x, self.right_boundaries_y):
			# Check if the point is within the bounding box of the road segment
			for i in range(len(left_boundary_x) - 1):
				if min(left_boundary_x[i], right_boundary_x[i], left_boundary_x[i + 1], right_boundary_x[i + 1]) <= target_position[0] <= max(left_boundary_x[i], right_boundary_x[i], left_boundary_x[i + 1], right_boundary_x[i + 1]) and \
				min(left_boundary_y[i], right_boundary_y[i], left_boundary_y[i + 1], right_boundary_y[i + 1]) <= target_position[1] <= max(left_boundary_y[i], right_boundary_y[i], left_boundary_y[i + 1], right_boundary_y[i + 1]):
					return True
		return False

	def process_road_network(self):
		'''Return true if target_position is on the road
			- target_position: 2d-tuple (x,y)
		'''
		for segment in self.road_network:
			geometry = segment["geometry"]
			px_values = [point["px"] for point in geometry]
			py_values = [point["py"] for point in geometry]
			tx_values = [point["tx"] for point in geometry]
			ty_values = [point["ty"] for point in geometry]
			width = segment["geometry"][0]["width"]

			# Calculate perpendicular vectors to represent the road boundaries
			normal_x = np.array([-ty for ty in ty_values])
			normal_y = np.array([tx for tx in tx_values])

			norm = np.sqrt(normal_x**2 + normal_y**2)
			normal_x /= norm
			normal_y /= norm

			# Calculate road boundaries
			self.left_boundaries_x.append(px_values - (width / 2) * normal_x)
			self.left_boundaries_y.append(py_values - (width / 2) * normal_y)
			self.right_boundaries_x.append( px_values + (width / 2) * normal_x)
			self.right_boundaries_y.append( py_values + (width / 2) * normal_y)
	
	def plot_road(self):
		for left_boundary_x, left_boundary_y, right_boundary_x, right_boundary_y in \
			zip(self.left_boundaries_x, self.left_boundaries_y, self.right_boundaries_x, self.right_boundaries_y):
			# # Plot the road
			# if 'inner' in segment:
			# 	color = 'g'
			# elif 'outer' in segment:
			# 	color = 'r'
			# else:
			# 	color = 'black'
			self.ax.plot(left_boundary_x, left_boundary_y, color=colors[i%len(colors)])
			self.ax.plot(right_boundary_x, right_boundary_y, color=colors[i%len(colors)])

		plt.xlabel('X-axis')
		plt.ylabel('Y-axis')
		plt.title('Road Visualization')
		plt.axis('equal')
		plt.show()


	# DO NOT USE : debug function
	def on_click(self, event):
		pass
		# # Extract the clicked coordinates
		# x, y = event.xdata, event.ydata
		# print(f"prev_x={self.prev_x}, prev_y={self.prev_y}")
		# print(f"x={x}, y={y}")

		# res = self.are_points_on_same_segment(self.prev_x, self.prev_y, x, y, self.road_network)
		# print(f"res={res}")
		# self.prev_x = x
		# self.prev_y = y


	# DO NOT USE : debug function
	def are_points_on_same_segment(self, x1, y1, x2, y2, road_data):
		for segment in road_data:
			geometry = segment["geometry"]
			px_values = [point["px"] for point in geometry]
			py_values = [point["py"] for point in geometry]
			tx_values = [point["tx"] for point in geometry]
			ty_values = [point["ty"] for point in geometry]
			width = segment["geometry"][0]["width"]

			# Calculate perpendicular vectors to represent the road boundaries
			dx = np.array([-ty for ty in ty_values])
			dy = np.array([tx for tx in tx_values])

			# Normalize the vectors
			length = np.sqrt(dx**2 + dy**2)
			dx /= length
			dy /= length

			# Calculate road boundaries
			left_boundary_x = px_values - (width / 2) * dx
			left_boundary_y = py_values - (width / 2) * dy
			right_boundary_x = px_values + (width / 2) * dx
			right_boundary_y = py_values + (width / 2) * dy

			# Check if both points are within the bounding box of the road segment
			if (min(left_boundary_x) <= x1 <= max(right_boundary_x) and
					min(left_boundary_y) <= y1 <= max(right_boundary_y) and
					min(left_boundary_x) <= x2 <= max(right_boundary_x) and
					min(left_boundary_y) <= y2 <= max(right_boundary_y)):
				return True

		return False
	
	def get_transform(self, source_frame, target_frame):
		"""Update the lidar-to-camera transform matrix from the tf topic"""
		try:
			# It’s lookup_transform(target_frame, source_frame, …) !!!
			transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			print('error')
			return

		# Build the matrix elements
		rotation_message = transform.transform.rotation
		rotation_quaternion = np.asarray((rotation_message.w, rotation_message.x, rotation_message.y, rotation_message.z))
		rotation_matrix = quaternions.quat2mat(rotation_quaternion)
		translation_message = transform.transform.translation
		translation_vector = np.asarray((translation_message.x, translation_message.y, translation_message.z)).reshape(3, 1)
		
		# Build the complete transform matrix
		return np.concatenate((
			np.concatenate((rotation_matrix, translation_vector), axis=1),
			np.asarray((0, 0, 0, 1)).reshape((1, 4))
		), axis=0)

	def get_world_position(self,car_baslink = np.array([0, 0, 0])):
		baslink_to_map = self.get_transform("camera_forward_optical_frame", self.config["map"]["world-frame"])
		position = baslink_to_map @ np.concatenate((car_baslink, np.array([1]))).T
		return (position/position[3])[:3]
	
	def get_car_world_position(self, world_position):
		map_to_baslink = self.get_transform(self.config["map"]["world-frame"], "camera_forward_optical_frame")
		position = map_to_baslink @ np.concatenate((world_position, np.array([1]))).T
		return (position/position[3])[:3]
			

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <config-file>")
	else:
		with open(sys.argv[1], "r") as config_file:
			config = yaml.load(config_file, yaml.Loader)
		rospy.init_node("distances")
		node = MapHandler(config)
		rospy.spin()