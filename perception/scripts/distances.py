#!/usr/bin/env python3

import os
import sys
import time
import rospy

import cv2 as cv
import numpy as np
np.float = np.float64 # fix for https://github.com/eric-wieser/ros_numpy/issues/37
from threading import Lock

import tf2_ros
import ros_numpy
import transforms3d.quaternions as quaternions
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

from object_detector import ObjectDetector



DISTANCE_SCALE_MIN = 0
DISTANCE_SCALE_MAX = 160

class DistanceExtractor (object):
	def __init__(self, image_topic, camerainfo_topic, pointcloud_topic, output_folder, tf_rate=1):
		self.image_topic = image_topic
		self.camerainfo_topic = camerainfo_topic
		self.pointcloud_topic = pointcloud_topic
		self.output_folder = output_folder

		# Initialize the topic subscribers
		self.image_subscriber = rospy.Subscriber(self.image_topic, Image, self.callback_image)
		self.camerainfo_subscriber = rospy.Subscriber(self.camerainfo_topic, CameraInfo, self.callback_camerainfo)
		self.pointcloud_subscriber = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.callback_pointcloud)
		
		# Initialize the transformation listener
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		# At first everything is null, no image can be produces if one of those is still null
		self.image_frame = None
		self.pointcloud_frame = None
		self.latest_image = None
		self.latest_pointcloud = None
		self.lidar_to_camera = None
		self.camera_to_lidar = None
		self.distortion_parameters = None

		self.sensor_to_image = None

		rospy.loginfo("Everything ready")

	def update_transforms(self):
		"""Update the lidar-to-camera transform matrix from the tf topic"""
		try:
			# It’s lookup_transform(target_frame, source_frame, …) !!!
			transform = self.tf_buffer.lookup_transform(self.image_frame, self.pointcloud_frame, rospy.Time(0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			return

		# Build the matrix elements
		rotation_message = transform.transform.rotation
		rotation_quaternion = np.asarray((rotation_message.w, rotation_message.x, rotation_message.y, rotation_message.z))
		rotation_matrix = quaternions.quat2mat(rotation_quaternion)
		translation_message = transform.transform.translation
		translation_vector = np.asarray((translation_message.x, translation_message.y, translation_message.z)).reshape(3, 1)
		
		# Build the complete transform matrix
		self.lidar_to_camera = np.concatenate((
			np.concatenate((rotation_matrix, translation_vector), axis=1),
			np.asarray((0, 0, 0, 1)).reshape((1, 4))
		), axis=0)

	def callback_image(self, data):
		"""Extract an image from the camera"""
		self.image_frame = data.header.frame_id
		self.latest_image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))
		
		self.convert_pointcloud()

	def callback_pointcloud(self, data):
		"""Extract a point cloud from the lidar"""
		self.pointcloud_frame = data.header.frame_id

		# Extract the (x, y, z) points from the PointCloud2 
		points_3d = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data, remove_nans=True) # (N,3)

		# Convert to a matrix with points as columns, in homogeneous coordinates
		self.latest_pointcloud = np.concatenate((
			points_3d.transpose(),
			np.ones(points_3d.shape[0]).reshape((-1, points_3d.shape[0]))
		), axis=0) # (4,N)


	def callback_camerainfo(self, data):
		"""Update the camera info"""
		self.sensor_to_image = np.asarray(data.K).reshape((3, 3))
		self.distortion_parameters = data.D[0]

	def convert_pointcloud(self):
		"""Superimpose a point cloud from the lidar onto an image from the camera"""
		# Cannot work if any piece of information is missing
		if (self.image_frame is None or self.pointcloud_frame is None or
			self.latest_image is None or self.latest_pointcloud is None or
			self.sensor_to_image is None):
			return
		self.update_transforms()

		# Easier to make a gradient in HSV
		lidar_image = cv.cvtColor(self.latest_image, cv.COLOR_RGB2HSV)

		# HACK : Error in my code or error in the focal returned by /forwardCamera/camera_info …?
		sensor_to_image = self.sensor_to_image.copy()
		#camera_to_image[0, 0] *= 0.51
		#camera_to_image[1, 1] *= 0.51

		pointcloud = self.latest_pointcloud

		pointcloud_camera = self.lidar_to_camera @ pointcloud # (4,N) = (4,4) @ (4,N)


		# Filter out points that are behind the camera, otherwise the following calculations overlay them on the image with inverted coordinates
		valid_points = (pointcloud_camera[2, :] >= 0) # (N,)
		pointcloud_camera = pointcloud_camera[:, valid_points] # (4,N')
		pointcloud = pointcloud[:, valid_points] # (4,N')

		ro = np.linalg.norm(pointcloud_camera[:3,:], axis=0) # (N,)

		pointcloud_sensor = np.asarray((
			pointcloud_camera[0] / (pointcloud_camera[2]+(ro*self.distortion_parameters)),
			pointcloud_camera[1] / (pointcloud_camera[2]+(ro*self.distortion_parameters)),
			np.ones(pointcloud_camera.shape[1])
		))#.transpose().reshape(-1, 2) (3,N)

		pointcloud_image = sensor_to_image @ pointcloud_sensor # (3,N)

		# Finalize the projection ([u v w] ⟶ [u/w v/w])
		image_points = np.asarray((
			pointcloud_image[0] / (pointcloud_image[2]),
			pointcloud_image[1] / (pointcloud_image[2]),
		))#.transpose() # (2,N)

		# # Visualize the lidar data projection onto the image
		# temp_img = lidar_image.copy()
		# for i in range(image_points.shape[1]):
		# 	if 0 <= image_points[0,i] <= temp_img.shape[1] and 0 <= image_points[1,i] <= temp_img.shape[0]:
		# 		cv.drawMarker(temp_img, (int(image_points[0,i]), int(image_points[1,i])), (0, 255, 0), cv.MARKER_CROSS, 4)
			
		# cv.imshow('dist', temp_img)
		# cv.waitKey(5)


		# # Calculate the distance to each lidar point and the associated color gradient
		#distances = np.linalg.norm(pointcloud[:3,:], axis=0)
		colors = self.get_color(ro)

		# Write all points to the final image
		for color, point in zip(colors, image_points.T):
			if 0 <= point[0] < lidar_image.shape[1] and 0 <= point[1] < lidar_image.shape[0]:
				lidar_image[int(point[1]), int(point[0])] = color
				# `+` dots
				if point[1] > 0:
					lidar_image[int(point[1]) - 1, int(point[0])] = color
				if point[1] < lidar_image.shape[0] - 1:
					lidar_image[int(point[1]) + 1, int(point[0])] = color
				if point[0] > 0:
					lidar_image[int(point[1]), int(point[0]) - 1] = color
				if point[0] < lidar_image.shape[1] - 1:
					lidar_image[int(point[1]), int(point[0]) + 1] = color

		rgb_output_image = cv.cvtColor(lidar_image, cv.COLOR_HSV2BGR)
		cv.imshow('dist', rgb_output_image)
		cv.waitKey(5)

	def get_color(self, distances):
		# Generate the color gradient from point distances
		return np.asarray((
			(255 * ((distances - DISTANCE_SCALE_MIN) / (DISTANCE_SCALE_MAX - DISTANCE_SCALE_MIN))) % 255,
			255 * np.ones(distances.shape[0]),
			255 * np.ones(distances.shape[0]),
		)).astype(np.uint8).transpose()


	

if __name__ == "__main__":
	rospy.init_node("distances")
	node = DistanceExtractor("/forwardCamera/image_raw", "/forwardCamera/camera_info", "/lidar", '$HOME/Documents')
	rospy.spin()

