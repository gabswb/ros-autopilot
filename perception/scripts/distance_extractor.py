#!/usr/bin/env python3

import sys
import rospy
import yaml

import cv2 as cv
import numpy as np
np.float = np.float64 # fix for https://github.com/eric-wieser/ros_numpy/issues/37

import tf2_ros
import ros_numpy
from cv_bridge import CvBridge
import transforms3d.quaternions as quaternions

from sensor_msgs.msg import CameraInfo, Image
from perception.msg import Object, ObjectBoundingBox

from common_scripts import map_handler


DISTANCE_SCALE_MIN = 0
DISTANCE_SCALE_MAX = 160

class DistanceExtractor (object):
	def __init__(self, config, camera_info_topic, lidar_viz_topic, visualize_lidar=False, use_map=False):
		self.config = config
		self.use_map = use_map
		self.visualize_lidar = visualize_lidar
		self.map_frame = self.config["map"]["world-frame"]
		self.reference_frame = self.config["map"]["reference-frame"]

		if self.use_map:
			self.map_handler = map_handler.MapHandler(self.config)
		else:
			self.sensor_to_image = np.asarray([[1124.66943359375, 0.0, 505.781982421875],
											   [0.0, 1124.6165771484375, 387.8110046386719],
											   [0.0, 0.0, 1.0]]).reshape((3, 3))
			self.distortion_parameter = 0.8803200125694275

		# Initialize the topic subscribers
		self.camerainfo_subscriber = rospy.Subscriber(camera_info_topic, CameraInfo, self.callback_camerainfo)

		#Initialize the topic publishers
		self.cv_bridge = CvBridge()
		self.lidar_viz_publisher = rospy.Publisher(lidar_viz_topic, Image, queue_size=10)
		
		# Initialize the transformation listener
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.transforms = {}

		rospy.loginfo("DistanceExtractor initialized")


	def get_transform(self, source_frame, target_frame):
		"""Update the lidar-to-camera transform matrix from the tf topic"""

		if (source_frame, target_frame) in self.transforms:
			return self.transforms[(source_frame, target_frame)]
		
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
		transform_matrix = np.concatenate((
			np.concatenate((rotation_matrix, translation_vector), axis=1),
			np.asarray((0, 0, 0, 1)).reshape((1, 4))
			), axis=0)
		self.transforms[(source_frame, target_frame)] = transform_matrix
		return transform_matrix
	

	def image_preprocessing(self, data):
		"""Extract an image from the camera"""
		image_frame = data.header.frame_id
		image = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))
		
		return image_frame, image

	def pointcloud_preprocessing(self, data):
		"""Extract a point cloud from the lidar"""
		pointcloud_frame = data.header.frame_id

		# Extract the (x, y, z) points from the PointCloud2 
		points_3d = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data, remove_nans=True) # (N,3)

		# Convert to a matrix with points as columns, in homogeneous coordinates
		pointcloud =  np.concatenate((
			points_3d.transpose(),
			np.ones(points_3d.shape[0]).reshape((-1, points_3d.shape[0]))
		), axis=0) # (4,N)

		return pointcloud_frame, pointcloud


	def callback_camerainfo(self, data):
		"""Update the camera info"""
		if data.distortion_model.lower() != "mei":
			rospy.logerr(f"Bad distortion model : {data.distortion_model}")
			return
		
		self.sensor_to_image = np.asarray(data.K).reshape((3, 3))
		self.distortion_parameter = data.D[0]
		self.camerainfo_subscriber.unregister()

	def crop_bbox(self, bbox, scale):
		bbox_cropped = ObjectBoundingBox()
		x_crop = bbox.w//scale
		y_crop = bbox.h//scale
		bbox_cropped.x = bbox.x + x_crop
		bbox_cropped.y = bbox.y + y_crop
		bbox_cropped.w = bbox.w - x_crop*2
		bbox_cropped.h = bbox.h - y_crop*2
		bbox_cropped.class_id = bbox.class_id
		return bbox_cropped

	def get_objects_position(self, image_data, pointcloud_data, bbox_list):
		"""Superimpose a point cloud from the lidar onto an image from the camera"""

		if len(bbox_list) == 0: return []

		camera_frame, image = self.image_preprocessing(image_data)
		lidar_frame, pointcloud = self.pointcloud_preprocessing(pointcloud_data)

		sensor_to_image = self.sensor_to_image
		lidar_to_camera = self.get_transform(lidar_frame, camera_frame)
		lidar_to_reference = self.get_transform(lidar_frame, self.reference_frame)
		
		pointcloud_camera = lidar_to_camera @ pointcloud # (4,N) = (4,4) @ (4,N)
		pointcloud_reference = lidar_to_reference @ pointcloud # (4,N) = (4,4) @ (4,N)

		# Filter out points that are behind the camera, otherwise the following calculations overlay them on the image with inverted coordinates
		valid_points = (pointcloud_camera[2, :] >= 0) # (N,)
		pointcloud_camera = pointcloud_camera[:, valid_points] # (4,N')
		pointcloud_reference = pointcloud_reference[:, valid_points] # (4,N')

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

		if self.visualize_lidar:
			image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
			colors = self.get_color(ro)

			# Write all points to the final image
			for color, point in zip(colors, image_points.T):
				if 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]:
					cv.drawMarker(image, (int(point[0]), int(point[1])), (int(color[0]), int(color[1]), int(color[2])), cv.MARKER_CROSS, 4)

			image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
			self.lidar_viz_publisher.publish(self.cv_bridge.cv2_to_imgmsg(image))

		object_list = []

		for bbox in bbox_list:
			bbox_cropped = self.crop_bbox(bbox, 3)

			filter = ((bbox_cropped.x <= image_points[0]) & (image_points[0] <= bbox_cropped.x + bbox_cropped.w) &
						(bbox_cropped.y <= image_points[1]) & (image_points[1] <= bbox_cropped.y + bbox_cropped.h))
			#relevant_points_image = image_points[:, filter] # (2,N'')
			#relevant_points_camera = pointcloud_camera[:, filter] #(4,N'')
			relevant_points_reference = pointcloud_reference[:, filter]

			# distances in the reference frame 
			distances = np.linalg.norm(relevant_points_reference[:3,:], axis=0) #(N'',)
			if len(distances) == 0: continue # non lidar points projected onto the object

			#index = np.argsort(distances)[len(distances)//4]
			index = np.argmin(distances)
			distance = distances[index]
			position = relevant_points_reference[:, index]

			if self.use_map:
				# filter out object that are not on the road
				reference_to_map = self.get_transform(self.reference_frame, self.map_frame)
				position_map = reference_to_map @ position.T
				if not self.map_handler.is_on_road((position_map[0], position_map[2])):
					continue

			obj = Object()
			obj.bbox = bbox
			obj.distance = distance
			obj.x = position[0]
			obj.y = position[1]
			obj.z = position[2]
			obj.left_blink = False
			obj.right_blink = False
			object_list.append(obj)

		return object_list
	

	def get_color(self, distances):
		# Generate the color gradient from point distances
		return np.asarray((
			(255 * ((distances - DISTANCE_SCALE_MIN) / (DISTANCE_SCALE_MAX - DISTANCE_SCALE_MIN))) % 255,
			255 * np.ones(distances.shape[0]),
			255 * np.ones(distances.shape[0]),
		)).astype(np.uint8).transpose()
	


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <config-file> [--viz-lidar]")
	else:
		with open(sys.argv[1], "r") as config_file:
			config = yaml.load(config_file, yaml.Loader)
		rospy.init_node("distances")
		node = DistanceExtractor(config, '--viz-lidar' in sys.argv)
		rospy.spin()

