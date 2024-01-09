#!/usr/bin/env python3

import sys
import yaml
import numpy as np

import rospy
from geometry_msgs.msg import TwistStamped

from perception.msg import ObjectList
from decision.msg import CameraActivation

from common_scripts.map_handler import MapHandler, lane_side
from decision_state import DECISION_STATE
from controller import Controller

to_not_kill = (0,1,2,3,4,5,6,7,8,25)  # person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, umbrella
STOP_THRESHOLD_DISTANCE = 10  # meters
ADAPT_SPEED_DISTANCE = 9
DEFAULT_RATE = 100  # Hz
OVERTAKING_SPEED = 2
CRUISING_SPEED = 4
LEFT = -1
RIGHT = 1
CAR_LENGHT = 3
CAR_WIDTH = 2

scenario_1 = [
	(59.25447372, -47.2042948,    0.996886, CRUISING_SPEED),
	(68.74508458, -42.61156511,   1.03150283, CRUISING_SPEED),
	(74.53964479, -25.98481017,   1.21311468, CRUISING_SPEED),
	(0, 0, 0, 0),
]

class DecisionMaker(object):
	def __init__(self, config, publishing_rate):
		self.config = config

		# Topics
		self.velocity_topic = self.config["topic"]["velocity"]
		self.steering_angle = self.config["topic"]["steering-angle"]
		self.speed_topic = self.config["topic"]["speed"]
		self.speed_cap_topic = self.config["topic"]["speed-cap"]
		self.object_info_topic = self.config["topic"]["object-info"]
		self.toogle_navigation_service_name = self.config["service"]["toogle-navigation"]
		self.class_name_path = self.config["model"]["detection-model-class-names-path"]
		self.navigation = self.config["feature"]["navigation"]
		self.control_type = self.config["feature"]["controle-type"]
		self.control_ref_topic = self.config["topic"]["control-refs-topic"]
		self.camera_activation_topic = self.config["topic"]["camera-activation"]
		self.lane_width = self.config["map"]["lane-width"]
		self.start_overtaking_distance = 9  # meters

		# Situation states
		self.panic_mode = False
		self.real_speed = None
		self.real_angular_speed = None
		self.navigation
		self.object_list = []

		# Target states
		self.last_distance_to_target = None
		self.last_distance_to_target_time = None
		self.current_target_point = None
		self.current_target_speed = None
		self.target_reached_count = 0
		self.targets = scenario_1 # array of target points (x,y,z,speed)

		# Overtaking states
		self.overtaking = False
		self.overtaken_car = None
		self.overtaking_end_step = 0

		# Get classes names
		self.classes = None
		with open(self.class_name_path, "r") as f:
			self.classes = [line.strip() for line in f.readlines()]

		# Map handler
		self.map_handler = MapHandler(config)

		# Controller
		self.controller = Controller(config)
		self.camera_activation = rospy.Publisher(self.camera_activation_topic, CameraActivation, queue_size=10)

		# Initialize perception topic subscribers
		self.perception_subscriber = rospy.Subscriber(self.object_info_topic, ObjectList, self.callback_perception)

		# Initialize car topic subscribers
		self.velocity_subscriber = rospy.Subscriber(self.velocity_topic, TwistStamped, self.callback_velocity)
		rospy.loginfo("Decision ready")

	def callback_perception(self, data):
		"""Callback called to get the list of objects from the perception topic"""
		self.object_list = data.object_list

	def callback_velocity(self, data):
		"""Callback called to get the real speed and angular speed from the velocity topic"""
		velocity_msg = data.twist
		velocity_x = velocity_msg.linear.x
		velocity_y = velocity_msg.linear.y
		self.real_speed = np.linalg.norm([velocity_x, velocity_y])
		self.real_angular_speed = velocity_msg.angular.z

	def hazard_lights_on(self, object):
		return object.left_blink and object.right_blink

	def init_overtaking(self, preceding_object):
		self.overtaking = True
		self.overtaken_car = preceding_object
		self.overtaking_end_step = self.target_reached_count + 3
		self.activate_camera(forward_right=True)

		# overtaken car position (in car referential)
		overtaken_x, overtaken_y, overtaken_z = (
			self.overtaken_car.x,
			self.overtaken_car.y,
			self.overtaken_car.z,
		)

		# overtaking path points (in car referential)
		overtaking_target_points = np.array([
				[overtaken_x - self.lane_width, overtaken_y, overtaken_z],
				[overtaken_x - self.lane_width, overtaken_y, overtaken_z + CAR_LENGHT*2],
				[overtaken_x-self.lane_width/4, overtaken_y, overtaken_z + CAR_LENGHT * 4],
		])

		# overtaking path points (in world referential)
		overtaking_target_points = np.array([
				self.map_handler.get_world_position(overtaking_target_points[0]),
				self.map_handler.get_world_position(overtaking_target_points[1]),
				self.map_handler.get_world_position(overtaking_target_points[2]),
		])

		# insert overtaking target points in targets
		self.targets.insert(0, (self.current_target_point[0], self.current_target_point[1], self.current_target_point[2], self.current_target_speed))
		for i in range(2, 0, -1):
			self.targets.insert(0, (overtaking_target_points[i][0], overtaking_target_points[i][1], overtaking_target_points[i][2], OVERTAKING_SPEED))
		self.current_target_point = (overtaking_target_points[0][0], overtaking_target_points[0][1], overtaking_target_points[0][2])
		self.current_target_speed = OVERTAKING_SPEED

	def next_target(self):
		self.target_reached_count += 1
		target_tmp = self.targets.pop(0)
		self.current_target_point = target_tmp[0:3]
		self.current_target_speed = target_tmp[3]
		self.last_distance_to_target = None
		self.last_distance_to_target_time = None
		rospy.loginfo(f"ACTION: reach target (x={self.current_target_point[0]:.2f};y={self.current_target_point[1]:.2f}) at {self.current_target_speed} km/h")


	def get_direction(self, current_position):
		if self.current_target_point is None:
			self.next_target()

		# get current and target position
		target_x, target_y, target_z = self.current_target_point
		speed = self.current_target_speed
		x, y, z = current_position
		distance_to_target = np.linalg.norm([x - target_x, y - target_y])
		# rospy.loginfo(f"distance_to_target={distance_to_target:.2f}")

		# save the last distance to target every 2 seconds to check if we are going away from the target
		if (self.last_distance_to_target is None or self.last_distance_to_target_time is None):
			self.last_distance_to_target = distance_to_target
			self.last_distance_to_target_time = rospy.get_time()
		elif rospy.get_time() - self.last_distance_to_target_time > 1:
			if distance_to_target > self.last_distance_to_target:
				rospy.loginfo(f"INFO: going away from target")
				self.next_target()
			self.last_distance_to_target = distance_to_target
			self.last_distance_to_target_time = rospy.get_time()

		# if we reach the target point, go to the next one
		if distance_to_target < 1:
			self.next_target()

		# get target position in car referential
		target = self.map_handler.get_car_world_position(self.current_target_point)
		return target, speed

	def activate_camera(self, forward=True, forward_left=False, forward_right=False, backward=True):
		cam_act = CameraActivation()
		cam_act.forward = forward
		cam_act.forward_left = forward_left
		cam_act.forward_right = forward_right
		cam_act.backward = backward
		self.camera_activation.publish(cam_act)

	def decision_maker(self):
		# HANDLE PANIC MODE 
		if self.panic_mode: 
			rospy.loginfo(f"ACTION: STOP (PANIC MODE)")
			self.controller.handle_decision([0,0,0], 0, self.real_speed)
			return
		
		# HANDLE SCENARIO END
		if len(self.targets) == 0:
			rospy.loginfo(f"ACTION: STOP (SCENARIO IS DONE)")
			self.controller.handle_decision([0,0,0], 0, self.real_speed)
			return
		
		# GET CURRENT POSITION
		current_position = self.map_handler.get_world_position()
		# rospy.loginfo(f'{current_position}')
		# return
	
		objetct_on_left_line = []
		objetct_on_right_line = []
		preceding_object = None
		slowing_down = False
		stop = False

		# LOOK FOR SURROUNDING DANGERS
		if not self.overtaking:
			for obj in self.object_list:
				if obj.bbox.class_id in to_not_kill:
					obj_lane_side = self.map_handler.get_lane_side(obj)
					if obj_lane_side == lane_side.LEFT:
						rospy.loginfo(f"INFO: Left lane {obj.distance:.2f}m away from {self.classes[obj.bbox.class_id]}")
						objetct_on_left_line.append(obj)
					elif obj_lane_side == lane_side.RIGHT:
						rospy.loginfo(f"INFO: Right lane {obj.distance:.2f}m away from {self.classes[obj.bbox.class_id]}")
						objetct_on_right_line.append(obj)
					elif obj_lane_side == lane_side.FRONT:
						rospy.loginfo(f"INFO: Preceding {obj.distance:.2f}m away from {self.classes[obj.bbox.class_id]}")
						preceding_object = obj
						if obj.distance < STOP_THRESHOLD_DISTANCE:
							stop = True
						elif obj.distance < ADAPT_SPEED_DISTANCE * self.real_speed:
							slowing_down = True

		# START OVERTAKING
		if not self.overtaking and preceding_object is not None:
			if (self.real_speed < OVERTAKING_SPEED and self.hazard_lights_on(preceding_object) and preceding_object.z < self.start_overtaking_distance):
				rospy.loginfo("ACTION: start overtaking")
				self.init_overtaking(preceding_object)

		# CHECK IF OVERTAKING IS FINISHED
		if self.overtaking:
			if self.target_reached_count > self.overtaking_end_step:
				rospy.loginfo("INFO: overtaking ended")
				self.overtaking = False
				self.activate_camera(forward_right=False)

		# GET TARGET
		target_position, target_speed = self.get_direction(current_position)

		# HANDLE DANGER
		if slowing_down:
			rospy.loginfo(f"ACTION: SLOW DOWN")
			target_speed = self.real_speed/2
		if stop:
			rospy.loginfo(f"ACTION: STOP")
			target_speed = 0
		
		# SEND CONTROLS
		self.controller.handle_decision(target_position, target_speed, self.real_speed)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage : {sys.argv[0]} <parameter-file> <publish-rate (optional)>")
	else:
		with open(sys.argv[1], "r") as parameterfile:
			config = yaml.load(parameterfile, yaml.Loader)

		rospy.init_node("decision")
		publishing_rate = (float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else DEFAULT_RATE)
		rospy.loginfo(f"Publishing rate: {publishing_rate} Hz")
		node = DecisionMaker(config, publishing_rate)
		node.activate_camera(forward=True, forward_left=False, forward_right=False, backward=False)

		rate = rospy.Rate(publishing_rate)
		while not rospy.is_shutdown():
			node.decision_maker()
			rate.sleep()

		rospy.spin()
