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
from decision.msg import DecisionInfo
from road import Road

WINDOW_SIZE_X = 75
WINDOW_SIZE_Y = 75


class MapPlotter(object):
	def __init__(self, config):
		self.config = config

		with open(self.config["map"]["road-network-path"], 'r') as f:
			self.road_network = json.load(f)

		self.segment_dict = {}
		self.road_list = []
		self.path = []

		self.process_road_network()
		self.create_path()

		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.objects_position = []
		self.car_position = self.get_world_position(np.array([0, 0, 0, 1]))

		self.fig, self.ax = plt.subplots()
		self.ln, = plt.plot([], [], 'ro')
		self.obj_scatter = None

		# Decision info
		self.targets = []

	def update_plot(self, frame):
		self.car_position = self.get_world_position(np.array([0,0,0,1]))

		x = self.car_position[0]
		y = self.car_position[1]
		self.ax.set_xlim(x-WINDOW_SIZE_X//2, x+WINDOW_SIZE_X//2)
		self.ax.set_ylim(y-WINDOW_SIZE_Y//2, y+WINDOW_SIZE_Y//2)
		self.ax.set_aspect('equal')
		self.ln.set_data(x, y)
		
		if self.obj_scatter is not None and self.obj_scatter in self.ax.collections:
			self.obj_scatter.remove()
		if self.objects_position:
			obj_positions = np.array(self.objects_position)
			self.obj_scatter = self.ax.scatter(obj_positions[:, 0], obj_positions[:, 1], c='blue', marker='o', label='Objects')
		
		if len(self.targets) > 0:
			targets = np.array(self.targets)
			self.ax.scatter(targets[:, 0], targets[:, 1], c='red', marker='X', label='Targets')

		return self.ln, self.obj_scatter

	def perception_callback(self, data):
		self.objects_position = []
		print(len(data.object_list))
		for obj in data.object_list:
			obj_pos = self.get_world_position(np.array([obj.x, obj.y, obj.z, 1]))
			self.objects_position.append(obj_pos)

	def decision_callback(self, data):
		self.targets = []
		for target in [data.target1, data.target2, data.target3, data.target4, data.target5]:
			if len(target) > 0:
				self.targets.append(target)

	def assign_segment_to_road(self, segment, road_id):
		self.segment_dict[segment["guid"]] = road_id

	def is_segment_assigned(self, segment_guid):
		segments_assigned = list(self.segment_dict.keys())
		if segment_guid not in segments_assigned:
			return False
		else:
			return self.segment_dict[segment_guid]

	def get_segment(self, segment_guid):
		for segment in self.road_network:
			if segment["guid"] == segment_guid:
				return segment

	def get_road(self, segment):
		for road in self.road_list:
			if segment in road.segment_list:
				return road

	def plot_init(self):
		for i, road in enumerate(self.road_list):
			if road.display:
				x_left, y_left = zip(*road.left_points)
				x_middle, y_middle = zip(*road.path_to_follow)
				x_right, y_right = zip(*road.right_points)

				# Tracer le graphique
				self.ax.plot(x_left, y_left, linestyle='-', color='black')
				if road in self.path:
					if road == self.path[0]:
						self.ax.plot(x_middle, y_middle, linestyle='-', color='red')
					else:
						self.ax.plot(x_middle, y_middle, linestyle='-', color='blue')
				self.ax.plot(x_right, y_right, linestyle='-', color='black')

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
		for i in range(len(self.road_network)):
			if self.is_segment_assigned(self.road_network[i]["guid"]) is False:
				Road(self, self.road_network[i])

	def create_path(self):
		first_road = self.road_list[59]
		second_road = first_road.forwardRoad
		third_road = second_road.leftRoad
		fourth_road = third_road.forwardRoad

		self.path.append(first_road)
		self.path.append(second_road)
		self.path.append(third_road)
		self.path.append(fourth_road)
		

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <config-file>")
	else:
		with open(sys.argv[1], "r") as config_file:
			config = yaml.load(config_file, yaml.Loader)
		rospy.init_node("map_plotter")
		node = MapPlotter(config)
		perception_subscriper = rospy.Subscriber(config["topic"]["object-info"], ObjectList, node.perception_callback)
		decision_subscriper = rospy.Subscriber(config["topic"]["decision-info"], DecisionInfo, node.decision_callback)
		ani = FuncAnimation(node.fig, node.update_plot, init_func=node.plot_init)
		plt.show(block=True)  # Display the initial plot
		rospy.spin()