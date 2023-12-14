#!/usr/bin/env python3

import sys
import yaml
import numpy as np

import rospy
from std_msgs.msg import Float32, String, UInt8
from std_srvs.srv import Empty
from geometry_msgs.msg import TwistStamped
from perception.msg import ObjectList, Object, ObjectBoundingBox


to_not_kill = (0,1,2,3,4,5,6,7,8, 25) #person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, umbrella
STOP_THRESHOLD_DISTANCE = 10 # meters
RATE = 10 # Hz

LANE_WIDTH = 1.5
OPPOSITE_TRESHOLD_DISTANCE = 40
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

        # Situation states
        self.real_speed = None
        self.real_angular_speed = None
        self.object_list = []

        # Overtaking states
        self.overtaking = False
        self.start_overtaking_time = None
        self.overtaking_direction = -1

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
        rospy.wait_for_service(self.toogle_navigation_service_name)
        self.toogle_navigation_service = rospy.ServiceProxy(self.toogle_navigation_service_name, Empty)
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
    
    def overtake(self, direction = 1):
        if not self.overtaking:
            self.toogle_navigation()
            rospy.loginfo("Start overtaking")
            self.start_overtaking_time = rospy.get_time()
            self.overtaking = True
            self.overtaking_direction = direction
        else:
            gap = rospy.get_time() - self.start_overtaking_time
            #rospy.loginfo(f"Gap: {gap:.2f}")
            self.speed_publisher.publish(2)
            if gap < 2:
                self.steering_angle_publisher.publish(0)
            elif gap < 5:
                self.steering_angle_publisher.publish(self.overtaking_direction*20)
            elif gap < 7:
                self.steering_angle_publisher.publish(-self.overtaking_direction*20)
            elif gap < 10:
                self.steering_angle_publisher.publish(0)
            elif gap < 12:
                self.steering_angle_publisher.publish(-self.overtaking_direction*20)
            elif gap < 15:
                self.steering_angle_publisher.publish(self.overtaking_direction*20)
            elif gap < 17:
                self.steering_angle_publisher.publish(0)
            else:
                self.overtaking = False
                self.toogle_navigation()
                rospy.loginfo("Finished overtaking")
    
    def hazard_lights_on(self, object):
        return object.left_blink and object.right_blink

    def is_preceding(self, object):
        return (np.abs(object.x) <= LANE_WIDTH) and object.z >= 0

    def is_opposite(self, object):
        return (np.abs(object.x) >= LANE_WIDTH and np.abs(object.x) <= 2*LANE_WIDTH  and object.distance < OPPOSITE_TRESHOLD_DISTANCE)

    def toogle_navigation(self,):
        try:
            _ = node.toogle_navigation_service()
        except rospy.ServiceException as e:
            pass

    def publish_control_inputs(self, ):
        """Publish the control inputs to the car"""
        # STOP CAR IF OBJECT IS TOO CLOSE
        opposite_objects = []
        preceding_object = None
        if not self.overtaking:
            for obj in self.object_list:
                if obj.bbox.class_id in to_not_kill:
                    if self.is_opposite(obj):
                        rospy.loginfo(f"Opposite {obj.distance:.2f}m away from {self.classes[obj.bbox.class_id]}")
                        opposite_objects.append(obj)
                    if self.is_preceding(obj):
                        rospy.loginfo(f"Preceding {obj.distance:.2f}m away from {self.classes[obj.bbox.class_id]}")
                        preceding_object = obj
                        if (obj.distance < STOP_THRESHOLD_DISTANCE*self.real_speed or obj.distance < STOP_THRESHOLD_DISTANCE):
                            rospy.loginfo(f"STOP")
                            self.speed_publisher.publish(0)

        # START OVERTAKING FROM LEFT
        if preceding_object is not None:
            if self.real_speed < 1 and self.hazard_lights_on(preceding_object):
                if len(opposite_objects) == 0: 
                    self.overtake(LEFT)
                else:
                    rospy.loginfo('Opposite object => not overtaking')
        
        # CONTINUE OVERTAKING IF 
        if self.overtaking:
            self.overtake()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <parameter-file>")
    else:
        with open(sys.argv[1], "r") as parameterfile:
            config = yaml.load(parameterfile, yaml.Loader)

        rospy.init_node("decision")
        node = Controller(config)
        node.toogle_navigation()
        rate = rospy.Rate(RATE)
        while not rospy.is_shutdown():
            node.publish_control_inputs()
            rate.sleep()

        rospy.spin()
