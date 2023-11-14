#!/usr/bin/env python3

import os
import sys
import time
import rospy
import yaml

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
	def __init__(self, config):
		self.config = config

		self.image_topic = self.config["node"]["image-topic"]
		self.camerainfo_topic = self.config["node"]["camerainfo-topic"]
		self.pointcloud_topic = self.config["node"]["pointcloud-topic"]

		# Object detector
		self.object_detector = ObjectDetector(self.config)

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
		self.distortion_parameter = None

		rospy.loginfo("DistanceExtractor initialized")

	def get_transform(self, source_frame, target_frame):
		"""Update the lidar-to-camera transform matrix from the tf topic"""
		try:
			# It’s lookup_transform(target_frame, source_frame, …) !!!
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
		a = np.concatenate((
			np.concatenate((rotation_matrix, translation_vector), axis=1),
			np.asarray((0, 0, 0, 1)).reshape((1, 4))
		), axis=0)
		# rospy.loginfo(f'lidar_to_camera.shape {a.shape}')
		# rospy.loginfo(f'lidar_to_camera {a}')
		return a
	

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
		self.distortion_parameter = data.D[0]

	def convert_pointcloud(self):
		"""Superimpose a point cloud from the lidar onto an image from the camera"""

		image = self.latest_image
		objects_bbox = self.object_detector.detect(image)

		if len(objects_bbox) > 0:
			pointcloud = self.latest_pointcloud
			sensor_to_image = self.sensor_to_image
			lidar_to_camera = self.get_transform(self.pointcloud_frame, self.image_frame)

			# Easier to make a gradient in HSV
			# image_to_print = cv.cvtColor(image, cv.COLOR_RGB2HSV)

			pointcloud_camera = lidar_to_camera @ pointcloud # (4,N) = (4,4) @ (4,N)

			# Filter out points that are behind the camera, otherwise the following calculations overlay them on the image with inverted coordinates
			valid_points = (pointcloud_camera[2, :] >= 0) # (N,)
			pointcloud_camera = pointcloud_camera[:, valid_points] # (4,N')
			pointcloud = pointcloud[:, valid_points] # (4,N')


			ro = np.linalg.norm(pointcloud_camera[:3,:], axis=0) # (N',)

			pointcloud_sensor = np.asarray((
				pointcloud_camera[0] / (pointcloud_camera[2]+(ro*self.distortion_parameter)),
				pointcloud_camera[1] / (pointcloud_camera[2]+(ro*self.distortion_parameter)),
				np.ones(pointcloud_camera.shape[1])
			))

			pointcloud_image = sensor_to_image @ pointcloud_sensor # (3,N')

			# Finalize the projection ([u v w] ⟶ [u/w v/w])
			image_points = np.asarray((
				pointcloud_image[0] / (pointcloud_image[2]),
				pointcloud_image[1] / (pointcloud_image[2]),
			)) # (2,N')

			for bbox in objects_bbox:
				bbox_cropped= bbox
				x_crop = bbox.w//3
				y_crop = bbox.h//3
				bbox_cropped.x += x_crop
				bbox_cropped.y += y_crop
				bbox_cropped.w -= x_crop*2
				bbox_cropped.h -= y_crop*2
				filter = ((bbox_cropped.x <= image_points[0]) & (image_points[0] <= bbox_cropped.x + bbox_cropped.w) &
			  			  (bbox_cropped.y <= image_points[1]) & (image_points[1] <= bbox_cropped.y + bbox_cropped.h))
				relevant_points_image = image_points[:, filter] # (2,N'')
				relevant_points_camera = pointcloud_camera[:, filter] #(4,N'')

				# distances = ro[filter]
				# distances in the camera frame 
				distances = np.linalg.norm(relevant_points_camera[:3,:], axis=0) #(N'',)
				#colors = self.get_color(distances)

				index = np.argsort(distances)[len(distances)//2]
				dist = distances[index]

				self.object_detector.draw_bounding_box(image, bbox.class_id, 0.5, bbox.x, bbox.y, bbox.x+bbox.w, bbox.y+bbox.h)
				image = cv.putText(image, f'd = {dist} m', (bbox.x, bbox.y-25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
			cv.imshow('dist', image)
			cv.waitKey(5)

			# output_img = image.copy()
			# # Write all points to the final image
			# for color, point in zip(colors, image_points.T):
			# 	if 0 <= point[0] < output_img.shape[1] and 0 <= point[1] < output_img.shape[0]:
			# 			cv.drawMarker(output_img, (int(point[0]), int(point[1])), color, cv.MARKER_CROSS, 4)


			# # rgb_output_image = cv.cvtColor(image_to_print, cv.COLOR_HSV2BGR)
			# cv.imshow('dist', output_img)
			# cv.waitKey(5)

	def get_color(self, distances):
		# Generate the color gradient from point distances
		return np.asarray((
			(255 * ((distances - DISTANCE_SCALE_MIN) / (DISTANCE_SCALE_MAX - DISTANCE_SCALE_MIN))) % 255,
			255 * np.ones(distances.shape[0]),
			255 * np.ones(distances.shape[0]),
		)).astype(np.uint8).transpose()


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <config-file>")
	else:
		with open(sys.argv[1], "r") as config_file:
			config = yaml.load(config_file, yaml.Loader)
	rospy.init_node("distances")
	node = DistanceExtractor(config)
	rospy.spin()

