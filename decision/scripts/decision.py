#!/usr/bin/env python3

import sys
import yaml
import numpy as np

import rospy
from std_msgs.msg import Float32, String, UInt8
from geometry_msgs.msg import TwistStamped
from perception.msg import ObjectList, Object, ObjectBoundingBox


to_not_kill = (0,1,2,3,4,5,6,7,8, 25) #person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, umbrella
THRESHOLD_DISTANCE = 10 # meters
RATE = 10 # Hz

class Controller(object):
    def __init__(self, config):
        self.config = config

        # Topics
        self.velocity_topic = self.config["topic"]["velocity"]
        self.steering_angle = self.config["topic"]["steering-angle"]
        self.speed_topic = self.config["topic"]["speed"]
        self.speed_cap_topic = self.config["topic"]["speed-cap"]
        self.object_info_topic = self.config["topic"]["object-info"]
        self.class_name_path = self.config["model"]["detection-model-class-names-path"]

        # Situation states
        self.real_speed = None
        self.real_angular_speed = None
        self.object_list = []

        # Overtaking states
        self.overtaking = False
        self.start_overtaking_time = None
        self.has_overtaken = False

        # Get classes names
        self.classes = None
        with open(self.class_name_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Initialize perception topic subscribers
        self.perception_subscriber = rospy.Subscriber(self.object_info_topic, ObjectList, self.callback_perception)

        # Initialize car topic subscribers
        self.velocity_subscriber = rospy.Subscriber(self.velocity_topic, TwistStamped, self.callback_velocity)

        # Initialize the car control publishers
        self.speed_publisher = rospy.Publisher(self.speed_topic, Float32, queue_size=10)
        self.steering_angle_publisher = rospy.Publisher(self.steering_angle, Float32, queue_size=10)
        rospy.loginfo("Decision ready")


    def callback_perception(self, data):
        """Callback called to get the list of objects from the perception topic"""
        # TODO not automatically remove list (add kalman filter or a confidence timing befor removing)
        self.object_list = data.object_list

    def callback_velocity(self, data):
        """Callback called to get the real speed and angular speed from the velocity topic"""
        # rospy.loginfo("Received a velocity")
        velocity_msg = data.twist
        velocity_x = velocity_msg.linear.x
        velocity_y = velocity_msg.linear.y
        self.real_speed = np.linalg.norm([velocity_x, velocity_y])
        self.real_angular_speed = velocity_msg.angular.z
    
    def overtake(self, side="left"):
        if side != "left" and side != "right":
            rospy.loginfo("Invalid side parameter for overtaking")
            return
        direction = -1 if side == "left"  else 1 

        if not self.overtaking:
            rospy.loginfo("Start overtaking")
            self.start_overtaking_time = rospy.get_time()
            self.overtaking = True
        else:
            gap = rospy.get_time() - self.start_overtaking_time
            rospy.loginfo(f"Gap: {gap:.2f}")
            self.speed_publisher.publish(2)
            if gap < 2:
                self.steering_angle_publisher.publish(0)
            elif gap < 5:
                self.steering_angle_publisher.publish(direction*20)
            elif gap < 7:
                self.steering_angle_publisher.publish(-direction*20)
            elif gap < 10:
                self.steering_angle_publisher.publish(0)
            elif gap < 12:
                self.steering_angle_publisher.publish(-direction*20)
            elif gap < 15:
                self.steering_angle_publisher.publish(direction*20)
            elif gap < 17:
                self.steering_angle_publisher.publish(0)
            else:
                self.overtaking = False
                rospy.loginfo("Finished overtaking")
        
    def has_hazard_lights_on(self, obj):
        """Check if the hazard lights are on for the given object."""
        return obj.left_blink or obj.right_blink

    def is_on_the_same_lane(self, obj, max_lateral_distance=3):
        """Check if the object is on the same lane as the car."""
        return abs(obj.bbox.y) <= max_lateral_distance

    def vehicule_around(self, target_obj):
        if not target_obj:
            return False
        # We can add more conditions here to determine if an object will be a problem for overtaking
        return self.is_on_the_same_lane(target_obj)

    def publish_control_inputs(self, ):
        """Publish the control inputs to the car"""
        object_in_front = None
        # STOP CAR IF OBJECT IS TOO CLOSE
        if len(self.object_list) > 0:
            for obj in self.object_list:
                if obj.bbox.class_id in to_not_kill:
                    rospy.loginfo(f"{obj.distance:.2f}m away from {self.classes[obj.bbox.class_id]}")
                    if obj.distance < THRESHOLD_DISTANCE:
                        rospy.loginfo(f"STOP")
                        self.speed_publisher.publish(0)
                        object_in_front = obj

        # OVERTAKE CHECK
        if object_in_front:
            if self.has_hazard_lights_on(object_in_front):
                no_vehicles_around = True
                for obj in self.object_list:
                    if obj != object_in_front and self.vehicule_around(obj):
                        no_vehicles_around = False
                        break 
                    
                if no_vehicles_around:
                    rospy.loginfo("Overtaking allowed")
                    self.overtake("left")
                    if self.overtaking == False:
                        self.has_overtaken = True
                else:
                    rospy.loginfo("Other vehicles around. Cannot overtake")
                    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <parameter-file>")
    else:
        with open(sys.argv[1], "r") as parameterfile:
            config = yaml.load(parameterfile, yaml.Loader)

        rospy.init_node("decision")
        node = Controller(config)
        rate = rospy.Rate(RATE)
        while not rospy.is_shutdown():
            node.publish_control_inputs()
            rate.sleep()

        rospy.spin()
