#!/usr/bin/env python3

import sys
import yaml
import numpy as np
np.float = np.float64 # fix for https://github.com/eric-wieser/ros_numpy/issues/37
import transforms3d.quaternions as quaternions
from sklearn.neighbors import KernelDensity

import rospy
import tf2_ros
import ros_numpy
import fish2bird
import cv2

from sensor_msgs.msg import CameraInfo
from perception.msg import Object, ObjectBoundingBox


DISTANCE_SCALE_MIN = 0
DISTANCE_SCALE_MAX = 160

class DistanceExtractor (object):
	def __init__(self, config):
		self.config = config
		self.camerainfo_topic = self.config["node"]["camerainfo-topic"]

		# Initialize the topic publisher
		self.status_seq = 0
		
		# Initialize the transformation listener
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		# At first everything is null, no image can be produces if one of those is still null
		self.image_frame = None
		self.pointcloud_frame = None
		self.latest_image = None
		self.latest_pointcloud = None
		self.pointcloud_array = []
		self.pointcloud_stamp_array = []
		self.lidar_to_camera = None
		self.lidar_to_baselink = None
		self.distortion_parameters = None
		self.camera_to_image = None
		self.image_stamp = None
		self.pointcloud_stamp = None

		# Initialize the topic subscribers
		self.camerainfo_subscriber = rospy.Subscriber(self.camerainfo_topic, CameraInfo, self.callback_camerainfo)

		rospy.loginfo("Distance extractor ready")	

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
		return np.concatenate((
			np.concatenate((rotation_matrix, translation_vector), axis=1),
			np.asarray((0, 0, 0, 1)).reshape((1, 4))
		), axis=0)

	def callback_image(self, img_data):
		"""Extract an image from the camera"""
		self.image_frame = img_data.header.frame_id
		self.image_stamp = img_data.header.stamp
		self.latest_image = np.frombuffer(img_data.data, dtype=np.uint8).reshape((img_data.height, img_data.width, 3))


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

	def callback_camerainfo(self, data):
		"""Callback called when a new camera info message is published"""
		# fish2bird only supports the camera model defined by Christopher Mei
		if data.distortion_model.lower() != "mei":
			rospy.logerr(f"Bad distortion model : {data.distortion_model}")
			return
		self.camera_to_image = np.asarray(data.P).reshape((3, 4))
		self.distortion_parameters = data.D
		self.camerainfo_subscriber.unregister()

	def lidar_to_image(self, pointcloud): 
		return fish2bird.target_to_image(pointcloud, self.lidar_to_camera, self.camera_to_image, self.distortion_parameters[0])
	
	def project_lidar_to_image(self, img, img_data, point_cloud_data):
		if (self.camera_to_image is None):
			raise Exception("Camera info not received yet")

		self.callback_image(img, img_data)
		self.callback_pointcloud(point_cloud_data)

		self.lidar_to_camera = self.get_transform(self.pointcloud_frame, self.image_frame)
		self.lidar_to_baselink = self.get_transform(self.pointcloud_frame, self.config["node"]["road-frame"])
		pointcloud = np.ascontiguousarray(self.pointcloud_array[0])

		self.pointcloud_array = []
		self.pointcloud_stamp_array = []

		lidar_coordinates_in_image = self.lidar_to_image(pointcloud)
		camera_pointcloud = self.lidar_to_camera @ pointcloud

		# Visualize the lidar data projection onto the image
		for i, point in enumerate(lidar_coordinates_in_image.T):
				# Filter out points that are not in the image dimension or behind the camera
				if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0] and camera_pointcloud[2, i] >=0:
					cv2.drawMarker(img, (int(point[0]), int(point[1])), (0, 255, 0), cv2.MARKER_CROSS, 4)

	def get_objects_position(self, img_data, point_cloud_data, bbox_list):
		"""Superimpose a point cloud from the lidar onto an image from the camera and publish the distance to the traffic sign bounding box on the topic"""
		if (self.camera_to_image is None):
			raise Exception("Camera info not received yet")

		self.callback_image(img_data)
		self.callback_pointcloud(point_cloud_data)

		self.lidar_to_camera = self.get_transform(self.pointcloud_frame, self.image_frame)
		self.lidar_to_baselink = self.get_transform(self.pointcloud_frame, self.config["node"]["road-frame"])
		pointcloud = np.ascontiguousarray(self.pointcloud_array[0])
		self.pointcloud_array = []
		self.pointcloud_stamp_array = []

		lidar_coordinates_in_image = self.lidar_to_image(pointcloud)

		obj_list = []

		for bbox in bbox_list:
			obj = Object()
			obj.bbox = bbox
			relevant_points_filter = ((obj.bbox.x <= lidar_coordinates_in_image[0]) & (lidar_coordinates_in_image[0] <= obj.bbox.x + obj.bbox.w) &
										(obj.bbox.y <= lidar_coordinates_in_image[1]) & (lidar_coordinates_in_image[1] <= obj.bbox.y + obj.bbox.h))
			relevant_points = pointcloud[:, relevant_points_filter]
			
			# We can still publish that we’ve seen it just in case, but we have no information on its position whatsoever
			if relevant_points.shape[1] == 0:
				obj.x = np.nan
				obj.y = np.nan
				obj.z = np.nan
			else:
				# Maximum density estimation to disregard the points that might be in the hitbox but physically behind the sign
				baselink_points = self.lidar_to_baselink @ relevant_points
				density_model = KernelDensity(kernel="epanechnikov", bandwidth=np.linalg.norm([obj.bbox.w, obj.bbox.h]) / 2)
				density_model.fit(baselink_points.T)
				point_density = density_model.score_samples(baselink_points.T)

				index = np.argmax(point_density)
				position_estimate = baselink_points[:, index]
				obj.x = position_estimate[0]
				obj.y = position_estimate[1]
				obj.z = position_estimate[2]
			
			obj_list.append(obj)

		return obj_list


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <config-file>")
	else:
		with open(sys.argv[1], "r") as config_file:
			config = yaml.load(config_file, yaml.Loader)
		rospy.init_node("perception_distances")
		node = DistanceExtractor(config)
		rospy.spin()
