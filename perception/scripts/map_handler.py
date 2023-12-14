import sys
import json
import yaml
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

import rospy

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

		# self.prev_x = 0
		# self.prev_y = 0
		# self.fig, self.ax = plt.subplots()
		# self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

		# self.plot_road()
		
	def get_lane_side(self, object):
		if np.abs(object.x) <= self.lane_width/2 and object.z >= 0:
			return lane_side.FRONT.value
		elif object.x < -self.lane_width/2 and object.x >= -self.lane_width*3/2:
			return lane_side.LEFT.value
		elif object.x > self.lane_width/2 and object.x <= self.lane_width*3/2:
			return lane_side.RIGHT.value
		else:
			return lane_side.AWAY.value
		
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
			

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <config-file>")
	else:
		with open(sys.argv[1], "r") as config_file:
			config = yaml.load(config_file, yaml.Loader)
		rospy.init_node("distances")
		node = MapHandler(config)
		rospy.spin()