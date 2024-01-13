#!/usr/bin/env python3

import sys
import yaml
import numpy as np

import rospy
from std_msgs.msg import Float32
from std_srvs.srv import Empty

STOP_THRESHOLD_DISTANCE = 10 # meters
DEFAULT_RATE = 100 # Hz

LANE_WIDTH = 1.5
LEFT = -1
RIGHT = 1

class Controller(object):
    def __init__(self, config):
        self.config = config

        # Topics
        self.velocity_topic = self.config["topic"]["velocity"]
        self.steering_angle = self.config["topic"]["steering-angle"]
        self.speed_topic = self.config["topic"]["speed"]
        self.speed_cap_topic = self.config["topic"]["speed-cap"]
        self.object_info_topic = self.config["topic"]["object-info"]
        self.toogle_navigation_service_name = self.config["service"]["toogle-navigation"]
        self.class_name_path = self.config["model"]["detection-model-class-names-path"]
        self.navigation_available = self.config["feature"]["navigation"]

        # Situation states
        self.real_speed = None
        self.real_angular_speed = None
        self.navigation = False
        self.last_steering_angle = 0
        self.object_list = []

        # Overtaking states
        self.overtaking = False
        self.start_overtaking_time = None
        self.overtaking_direction = -1

        # pure pursuit parameters
        self.k = 0.1
        self.LFC = 5
        self.min_ld = 2
        self.max_ld = 5
        self.L = 2.588  # distance between wheels

        # Initialize the car control publishers
        self.speed_publisher = rospy.Publisher(self.speed_topic, Float32, queue_size=10)
        self.steering_angle_publisher = rospy.Publisher(self.steering_angle, Float32, queue_size=10)

        if self.navigation_available:
            rospy.wait_for_service(self.toogle_navigation_service_name)
            self.toogle_navigation_service = rospy.ServiceProxy(self.toogle_navigation_service_name, Empty)
        rospy.loginfo("Controller ready")

    def publish_controls(self, speed, steering_angle):
        if speed is None or steering_angle is None:
            raise Exception("Speed or steering angle is None")
        self.speed_publisher.publish(speed)
        self.steering_angle_publisher.publish(steering_angle)

    def pure_pursuit_controls(self, target_position, speed):
        if target_position is not None:
            speed = speed or 1 # sometimes speed is None
            # l_d = np.linalg.norm([target_position[0], target_position[2]])
            l_d = self.k * speed + self.LFC
            alpha = np.arctan2(target_position[0], target_position[2])
            steering_angle = np.arctan2(2 * self.L * np.sin(alpha), l_d)
            self.last_steering_angle = steering_angle * 180 / np.pi
        return self.last_steering_angle


    def toogle_navigation(self,):
        if self.navigation_available:
            try:
                self.toogle_navigation_service()
            except rospy.ServiceException as e:
                pass
        else:
            self.publish_controls(speed=3,steering_angle=0) # FIXME useless
            
    def handle_decision(self, target_position, target_speed, current_speed = None):            
        steering_angle = self.pure_pursuit_controls(target_position, current_speed)
        self.publish_controls(steering_angle=steering_angle, speed=target_speed)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <parameter-file> <publish-rate (optional)>")
    else:
        with open(sys.argv[1], "r") as parameterfile:
            config = yaml.load(parameterfile, yaml.Loader)

        rospy.init_node("controller")
        publishing_rate = (float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else DEFAULT_RATE)
        rospy.loginfo(f"Publishing rate: {publishing_rate} Hz")

        node = Controller(config)
        rate = rospy.Rate(publishing_rate)

        while not rospy.is_shutdown():
            node.handle_decision([-2, 2], 2)
            rate.sleep()

        rospy.spin()
