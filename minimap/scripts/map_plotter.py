#!/usr/bin/env python3

import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rospy
import tf2_ros
import transforms3d.quaternions as quaternions

from perception.msg import ObjectList, Object, ObjectBoundingBox

WINDOW_SIZE_X = 50
WINDOW_SIZE_X = 50


class MapPlotter(object):
	def __init__(self, config):
		self.config = config

		with open(self.config["map"]["road-network-path"], 'r') as f:
				self.road_network = json.load(f)
		
		self.left_boundaries_x = []
		self.left_boundaries_y = []
		self.right_boundaries_x = []
		self.right_boundaries_y = []

		self.process_road_network()
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.objects_position = []
		self.car_position = self.get_world_position(np.array([0, 0, 0, 1]))

		self.fig, self.ax = plt.subplots()
		self.ln, = plt.plot([], [], 'ro')
		self.obj_scatter = None


	def update_plot(self, frame):

		x = self.car_position[0]
		y = self.car_position[1]
		self.ax.set_xlim(x-WINDOW_SIZE_X//2, x+WINDOW_SIZE_X//2)
		self.ax.set_ylim(y-WINDOW_SIZE_X//2, y+WINDOW_SIZE_X//2)
		self.ax.set_aspect('equal')
		self.ln.set_data(x, y)
		
		if self.obj_scatter is not None and self.obj_scatter in self.ax.collections:
			self.obj_scatter.remove()
		if self.objects_position:
			obj_positions = np.array(self.objects_position)
			self.obj_scatter = self.ax.scatter(obj_positions[:, 0], obj_positions[:, 1], c='blue', marker='o', label='Objects')


		return self.ln, self.obj_scatter

	def update_map(self, data):
		self.car_position = self.get_world_position(np.array([0,0,0,1]))

		self.objects_position = []
		print(len(data.object_list))
		for obj in data.object_list:
			obj_pos = self.get_world_position(np.array([obj.x, obj.y, obj.z, 1]))
			self.objects_position.append(obj_pos)
			

	def plot_init(self):
		for left_boundary_x, left_boundary_y, right_boundary_x, right_boundary_y in \
			zip(self.left_boundaries_x, self.left_boundaries_y, self.right_boundaries_x, self.right_boundaries_y):

			self.ax.plot(left_boundary_x, left_boundary_y, color='black')
			self.ax.plot(right_boundary_x, right_boundary_y, color='black')
		return self.ln  
		

	def get_transform(self, source_frame, target_frame):
		"""Update the lidar-to-camera transform matrix from the tf topic"""
		try:
			# It’s lookup_transform(target_frame, source_frame, …) !!!
			transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(3.0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
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

	def get_world_position(self, position):
		reference_to_map = self.get_transform(self.config["map"]["reference-frame"], self.config["map"]["world-frame"])
		return (reference_to_map @ position.T)[:3]




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
		

			

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <config-file>")
	else:
		with open(sys.argv[1], "r") as config_file:
			config = yaml.load(config_file, yaml.Loader)
		rospy.init_node("map_plotter")
		node = MapPlotter(config)
		sub = rospy.Subscriber(config["topic"]["object-info"], ObjectList, node.update_map)
		ani = FuncAnimation(node.fig, node.update_plot, init_func=node.plot_init)
		plt.show(block=True)  # Display the initial plot
		rospy.spin()