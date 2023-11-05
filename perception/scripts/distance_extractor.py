#!/usr/bin/env python3
import sys
import yaml
import numpy as np
np.float = np.float64 # fix for https://github.com/eric-wieser/ros_numpy/issues/37
import transforms3d.quaternions as quaternions

import rospy
import tf2_ros
import ros_numpy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

class DistanceExtractor (object):
	def __init__(self, config):
		self.config = config

		self.image_topic = self.config["node"]["image-topic"]
		self.camerainfo_topic = self.config["node"]["camerainfo-topic"]
		self.pointcloud_topic = self.config["node"]["pointcloud-topic"]

		self.pointcloud_subscriber = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.callback_pointcloud)

		
		# Initialize the transformation listener
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.distortion_parameters = None
		self.sensor2camera = None



	def callback_camerainfo(self, data):
		"""Callback called when a new camera info message is published"""
		# fish2bird only supports the camera model defined by Christopher Mei
		if data.distortion_model.lower() != "mei":
			rospy.logerr(f"Bad distortion model : {data.distortion_model}")
			return
		self.sensor2camera = np.asarray(data.P).reshape((3, 4))
		self.distortion_parameters = data.D


	def callback_image(self, data):
		"""Extract an image from the camera"""
		self.latest_image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))
			#cProfile.runctx("self.convert_pointcloud()", globals(), locals())


	def callback_pointcloud(self, data):
		"""Extract a point cloud from the lidar"""
		self.pointcloud_frame = data.header.frame_id
		self.pointcloud_stamp = data.header.stamp

		# Extract the (x, y, z) points from the PointCloud2 message
		points_3d = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data, remove_nans=True)

		# Convert to a matrix with points as columns, in homogeneous coordinates
		self.latest_pointcloud = np.concatenate((
			points_3d.transpose(),
			np.ones(points_3d.shape[0]).reshape((1, points_3d.shape[0]))
		), axis=0)

		self.pointcloud_stamp_array.append(self.pointcloud_stamp)
		self.pointcloud_array.append(self.latest_pointcloud)



	def get_transform(self, source_frame, target_frame):
		"""Update the lidar-to-camera transform matrix from the tf topic"""
		try:
			transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
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

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <config-file>")
	else:
		with open(sys.argv[1], "r") as config_file:
			config = yaml.load(config_file, yaml.Loader)
		rospy.init_node("perception_distances")
		node = DistanceExtractor(config)
		rospy.spin()
