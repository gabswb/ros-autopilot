#!/usr/bin/env python3

import sys
import yaml
import numpy as np

import rospy
from std_msgs.msg import Float32
from std_srvs.srv import Empty
from geometry_msgs.msg import TwistStamped

from perception.msg import ObjectList
from decision_state import DECISION_STATE


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
        self.object_list = []

        # Overtaking states
        self.overtaking = False
        self.start_overtaking_time = None
        self.overtaking_direction = -1

        # pure pursuit parameters
        self.k = 4
        self.min_ld = 1
        self.max_ld = 15
        self.L = 2.588 # distance between wheels

        # Get classes names
        self.classes = None
        with open(self.class_name_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Initialize perception topic subscribers
        rospy.Subscriber(self.object_info_topic, ObjectList, self.callback_perception)

        # Initialize car topic subscribers
        self.velocity_subscriber = rospy.Subscriber(self.velocity_topic, TwistStamped, self.callback_velocity)

        # Initialize the car control publishers
        self.speed_publisher = rospy.Publisher(self.speed_topic, Float32, queue_size=10)
        self.steering_angle_publisher = rospy.Publisher(self.steering_angle, Float32, queue_size=10)

        if self.navigation_available:
            rospy.wait_for_service(self.toogle_navigation_service_name)
            self.toogle_navigation_service = rospy.ServiceProxy(self.toogle_navigation_service_name, Empty)
        rospy.loginfo("Controller ready")

    def callback_perception(self, data):
        """Callback called to get the list of objects from the perception topic"""
        # TODO not automatically remove list (add kalman filter or a confidence timing befor removing)
        self.object_list = data.object_list

    def callback_velocity(self, data):
        """Callback called to get the real speed and angular speed from the velocity topic"""
        velocity_msg = data.twist
        velocity_x = velocity_msg.linear.x
        velocity_y = velocity_msg.linear.y
        self.real_speed = np.linalg.norm([velocity_x, velocity_y])
        self.real_angular_speed = velocity_msg.angular.z

    def publish_controls(self, speed = None, steering_angle = None):
        if speed is not None:
            self.speed_publisher.publish(speed)
        if steering_angle is not None:
            self.steering_angle_publisher.publish(steering_angle)

    def overtake(self, direction = -1):
        if not self.overtaking:
            rospy.loginfo("ACTION: START OVERTAKING")
            self.start_overtaking_time = rospy.get_time()
            self.overtaking = True
            self.overtaking_direction = direction
        else:
            gap = rospy.get_time() - self.start_overtaking_time
            #rospy.loginfo(f"Gap: {gap:.2f}")
            if gap < 2:
                self.publish_controls(steering_angle=0, speed=2)
            elif gap < 5:
                self.publish_controls(steering_angle=self.overtaking_direction*20, speed=2)
            elif gap < 7:
                self.publish_controls(steering_angle=-self.overtaking_direction*20, speed=2)
            elif gap < 10:
                self.publish_controls(steering_angle=0, speed=2)
            elif gap < 12:
                self.publish_controls(steering_angle=-self.overtaking_direction*20, speed=2)
            elif gap < 15:
                self.publish_controls(steering_angle=self.overtaking_direction*20, speed=2)
            elif gap < 17:
                self.publish_controls(steering_angle=0, speed=2)
            else:
                rospy.loginfo("INFO: overtaking finished")
                self.overtaking = False
                rospy.loginfo(f"ACTION: START NAVIGATION")
                self.toogle_navigation()

    def pure_pursuit_controls(self, target_position, speed):
        # l_d = np.clip(self.k * speed, self.min_ld, self.max_ld)
        l_d = np.linalg.norm(target_position)
        alpha = np.arctan2(target_position[0],target_position[1])
        steering_angle = np.arctan2(2 * self.L * np.sin(alpha), l_d)
        return steering_angle*180/np.pi


    def toogle_navigation(self,):
        if self.navigation_available:
            try:
                self.toogle_navigation_service()
            except rospy.ServiceException as e:
                pass
        else:
            self.publish_controls(speed=3,steering_angle=0) # FIXME useless
            
    def handle_decision(self, states:DECISION_STATE, target_position = None, speed = None):
        for state in states:
            if state == DECISION_STATE.STOP:
                rospy.loginfo(f"ACTION: STOP (object in front is too close)")
                self.publish_controls(speed=0, steering_angle=0)

            elif state == DECISION_STATE.ADAPT_TO_FORWARD_SPEED:
                rospy.loginfo(f"ACTION: SLOW DOWN (object in front is too close)")
                self.publish_controls(speed=0, steering_angle=0)
            
            elif state == DECISION_STATE.START_NAVIGATION:
                if self.navigation:
                    rospy.loginfo("INFO: navigation already started")
                else:
                    rospy.loginfo("ACTION: START NAVIGATION")
                    self.toogle_navigation()
                    self.navigation = True
            
            elif state == DECISION_STATE.STOP_NAVIGATION:
                if not self.navigation:
                    rospy.loginfo("INFO: navigation already stopped")
                else:
                    rospy.loginfo("ACTION: STOP NAVIGATION")
                    self.toogle_navigation()
                    self.navigation = False
            
            elif state == DECISION_STATE.OVERTAKE:
                # self.overtake(target_position)
                #target_position = [x,z]
                steering_angle = self.pure_pursuit_controls(target_position, speed)
                self.publish_controls(steering_angle=steering_angle, speed=2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <parameter-file> <publish-rate (optional)>")
    else:
        with open(sys.argv[1], "r") as parameterfile:
            config = yaml.load(parameterfile, yaml.Loader)

        rospy.init_node("controller")
        publishing_rate = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else DEFAULT_RATE
        rospy.loginfo(f"Publishing rate: {publishing_rate} Hz")
        
        node = Controller(config)
        rate = rospy.Rate(publishing_rate)

        while not rospy.is_shutdown():
            node.handle_decision([DECISION_STATE.OVERTAKE], [-2,2], 2)
            rate.sleep()

        rospy.spin()