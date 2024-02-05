#!/usr/bin/env python3

import sys
import yaml
import numpy as np
import math

import rospy
from geometry_msgs.msg import TwistStamped

from perception.msg import ObjectList
from decision.msg import CameraActivation, DecisionInfo

from common.map_handler import MapHandler, lane_side
from decision_state import DECISION_STATE
from controller import Controller

to_not_kill = (
0, 1, 2, 3, 4, 5, 6, 7, 8, 25)  # person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, umbrella
STOP_THRESHOLD_DISTANCE = 10  # meters
OVERTAKING_STOP_THRESHOLD_DISTANCE = 100  # meters
ADAPT_SPEED_DISTANCE = 9
DEFAULT_RATE = 100  # Hz
OVERTAKING_SPEED = 2
CRUISING_SPEED = 4
LEFT = -1
RIGHT = 1
CAR_LENGHT = 3
CAR_WIDTH = 2

class DecisionMaker(object):
    def __init__(self, config, publishing_rate):
        self.config = config

        # Topics
        self.velocity_topic = self.config["topic"]["velocity"]
        self.steering_angle = self.config["topic"]["steering-angle"]
        self.speed_topic = self.config["topic"]["speed"]
        self.speed_cap_topic = self.config["topic"]["speed-cap"]
        self.object_info_topic = self.config["topic"]["object-info"]
        self.decision_topic = self.config["topic"]["decision-info"]
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
        self.object_list = []

        # Map handler
        self.map_handler = MapHandler(config)

        # Path
        self.path = []
        self.create_path()
        self.targets = self.convert_path_to_target_points() # array of target points (x,y,z,speed)
        self.next_roads_to_check = self.path[:2]

        # Target states
        self.last_distance_to_target = None
        self.last_distance_to_target_time = None
        self.current_target_point = None
        self.current_target_speed = None
        self.target_reached_count = 0

        # Overtaking states
        self.overtaking = False
        self.overtaken_car = None
        self.overtaking_end_step = 0

        # Get classes names
        self.classes = None
        with open(self.class_name_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Controller
        self.controller = Controller(config)
        self.camera_activation = rospy.Publisher(self.camera_activation_topic, CameraActivation, queue_size=10)

        # Initialize decision topic publisher
        self.decision_publisher = rospy.Publisher(self.decision_topic, DecisionInfo, queue_size=10)

        # Initialize perception topic subscribers
        self.perception_subscriber = rospy.Subscriber(self.object_info_topic, ObjectList, self.callback_perception)

        # Initialize car topic subscribers
        self.velocity_subscriber = rospy.Subscriber(self.velocity_topic, TwistStamped, self.callback_velocity)
        self.next_target()
        rospy.loginfo("Decision ready")

    def create_path(self):
        first_road = self.map_handler.road_list[59]
        second_road = first_road.forwardRoad
        third_road = second_road.leftRoad
        fourth_road = third_road.forwardRoad
        fifth_road = fourth_road.leftRoad
        sixth_road = fifth_road.forwardRoad
        seventh_road = sixth_road.leftRoad

        self.path.append(first_road)
        self.path.append(second_road)
        self.path.append(third_road)
        self.path.append(fourth_road)
        self.path.append(fifth_road)
        self.path.append(sixth_road)
        self.path.append(seventh_road)

    def convert_path_to_target_points(self):
        previous_x, previous_y = 0, 0
        target_points = []
        for road in self.path[1:]:
            for point in road.path_to_follow[1:-1]:
                x, y = point
                x_round, y_round = round(x, 1), round(y, 1)
                if math.dist((previous_x, previous_y), (x_round, y_round)) > 2:
                    target_points.append((x_round, y_round, 1, CRUISING_SPEED))
                    previous_x, previous_y = x_round, y_round

        target_points.append((0, 0, 0, 0))
        return target_points

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
            [overtaken_x - self.lane_width, overtaken_y, overtaken_z + CAR_LENGHT * 2],
            [overtaken_x - self.lane_width / 4, overtaken_y, overtaken_z + CAR_LENGHT * 4],
        ])

        # overtaking path points (in world referential)
        overtaking_target_points = np.array([
            self.map_handler.get_world_position(overtaking_target_points[0]),
            self.map_handler.get_world_position(overtaking_target_points[1]),
            self.map_handler.get_world_position(overtaking_target_points[2]),
        ])

        # insert overtaking target points in targets
        self.targets.insert(0, (
        self.current_target_point[0], self.current_target_point[1], self.current_target_point[2],
        self.current_target_speed))
        for i in range(2, 0, -1):
            self.targets.insert(0, (
            overtaking_target_points[i][0], overtaking_target_points[i][1], overtaking_target_points[i][2],
            OVERTAKING_SPEED))
        self.current_target_point = (
        overtaking_target_points[0][0], overtaking_target_points[0][1], overtaking_target_points[0][2])
        self.current_target_speed = OVERTAKING_SPEED

    def next_target(self):
        self.target_reached_count += 1
        target_tmp = self.targets.pop(0)
        self.current_target_point = target_tmp[0:3]
        self.current_target_speed = target_tmp[3]
        self.last_distance_to_target = None
        self.last_distance_to_target_time = None
        rospy.loginfo(
            f"ACTION: reach target (x={self.current_target_point[0]:.2f};y={self.current_target_point[1]:.2f}) at {self.current_target_speed} km/h")

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
        if self.last_distance_to_target is None or self.last_distance_to_target_time is None:
            self.last_distance_to_target = distance_to_target
            self.last_distance_to_target_time = rospy.get_time()
        elif rospy.get_time() - self.last_distance_to_target_time > 0.5:
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

        # if target is behind the car, go to the next one
        if target[2] < 0:
            self.next_target()
            
        return target, speed

    def activate_camera(self, forward=True, forward_left=False, forward_right=False, backward=True):
        cam_act = CameraActivation()
        cam_act.forward = forward
        cam_act.forward_left = forward_left
        cam_act.forward_right = forward_right
        cam_act.backward = backward
        self.camera_activation.publish(cam_act)

    def publish_decision_info(self, roads_id, is_road_available):
        decision_info = DecisionInfo()
        decision_info.target1 = list(self.current_target_point)
        decision_info.target2 = list(self.targets[0][0:3]) if len(self.targets) > 1 else []
        decision_info.target3 = list(self.targets[1][0:3]) if len(self.targets) > 2 else []
        decision_info.target4 = list(self.targets[2][0:3]) if len(self.targets) > 3 else []
        decision_info.target5 = list(self.targets[3][0:3]) if len(self.targets) > 4 else []
        decision_info.target_speed = self.current_target_speed
        decision_info.roads_to_check_ids = roads_id
        decision_info.is_road_available = is_road_available
        self.decision_publisher.publish(decision_info)

    def decision_maker(self):
        # HANDLE PANIC MODE
        if self.panic_mode:
            rospy.loginfo(f"ACTION: STOP (PANIC MODE)")
            self.controller.handle_decision([0, 0, 0], 0, self.real_speed)
            return

        # HANDLE SCENARIO END
        if len(self.targets) == 0:
            rospy.loginfo(f"ACTION: STOP (SCENARIO IS DONE)")
            self.controller.handle_decision([0, 0, 0], 0, self.real_speed)
            return
        
        # PRINT CURRENT POSITION
        # rospy.loginfo(f'{current_position}')
        # return

        # RESET STATES
        current_position = self.map_handler.get_world_position()
        preceding_object = None
        slowing_down = False
        stop = False
        roads = []

        # DETERMINE NEXT ROAD 
        current_road = self.map_handler.get_road_position(current_position[:2], self.path)
        next_road = self.map_handler.get_road_position(self.current_target_point[:2], self.path)
        if not self.overtaking:
            i = 0
            while next_road == current_road:
                if i >= len(self.targets):
                    break
                next_road = self.map_handler.get_road_position(self.targets[i][:2], self.path)
                if next_road is None:
                    next_road = current_road
                i += 1

        # IF TARGET POINT ISN'T ON THE ROAD -> PANIC MODE 
        if next_road is None:
            if current_road is None:
                self.panic_mode = True
            return

        # DETERMINE ROADS TO CHECK
        roads_to_check = [next_road]

        if not self.overtaking:
            if current_road is not None:
                roads_to_check.append(current_road)

        roads_id = [road.id for road in roads_to_check]
        is_road_available = [2 for road in roads_to_check] # 0: STOP, 1: SLOW DOWN, 2: CLEAR

        # CHECK IF OBJECT ON ROADS TO CHECK
        for obj in self.object_list:
            x, y, z = self.map_handler.get_world_position([obj.x, obj.y, obj.z])
            object_road = self.map_handler.get_road_position((x, y))
            if object_road is not None:
                for i, road_to_to_check in enumerate(roads_to_check):
                    if road_to_to_check == object_road:
                        dist = np.linalg.norm([obj.x, obj.y, obj.z])
                        if obj.z > 0:
                            if self.overtaking and dist < OVERTAKING_STOP_THRESHOLD_DISTANCE:
                                stop = True
                                is_road_available[i] = 0
                            if dist < STOP_THRESHOLD_DISTANCE:
                                stop = True
                                is_road_available[i] = 0
                            elif dist < self.real_speed * ADAPT_SPEED_DISTANCE:
                                slowing_down = True
                                is_road_available[i] = 1

                            if preceding_object is None:
                                preceding_object = obj
                            elif obj.z < preceding_object.z:
                                preceding_object = obj

                roads.append(object_road)

        # START OVERTAKING
        if not self.overtaking and preceding_object is not None:
            if (self.real_speed < OVERTAKING_SPEED and self.hazard_lights_on(
                    preceding_object) and preceding_object.z < self.start_overtaking_distance):
                rospy.loginfo("ACTION: start overtaking")
                self.init_overtaking(preceding_object)

        # CHECK IF OVERTAKING IS FINISHED
        if self.overtaking:
            if self.target_reached_count > self.overtaking_end_step:
                rospy.loginfo("INFO: overtaking ended")
                self.overtaking = False
                self.activate_camera(forward_right=False)

        # GET TARGET TO FOLLOW
        target_position, target_speed = self.get_direction(current_position)

        # HANDLE DANGER (SLOW DOWN AND STOP)
        if slowing_down:
            rospy.loginfo(f"ACTION: SLOW DOWN")
            target_speed = self.real_speed / 2
        if stop:
            rospy.loginfo(f"ACTION: STOP")
            target_speed = 0

        # SEND CONTROLS
        self.controller.handle_decision(target_position, target_speed, self.real_speed)
        self.publish_decision_info(roads_id, is_road_available)
        self.next_roads_to_check = roads_to_check


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
