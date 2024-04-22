#!/usr/bin/env python3

import sys
import yaml
import numpy as np

import rospy
from std_msgs.msg import Float32, String, UInt8
from std_srvs.srv import Empty
from geometry_msgs.msg import TwistStamped
from perception.msg import ObjectList, Object, ObjectBoundingBox
from common.map_handler import MapHandler, lane_side


to_not_kill = (0,1,2,3,4,5,6,7,8, 25) #person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, umbrella
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
        self.navigation = self.config["feature"]["navigation"] 
        self.control_type = self.config["feature"]["controle-type"]
        self.control_ref_topic = self.config['topic']['control-refs-topic']

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

        # Map handler
        self.map_handler = MapHandler(config)

        # Initialize perception topic subscribers
        self.perception_subscriber = rospy.Subscriber(self.object_info_topic, ObjectList, self.callback_perception)

        # Initialize car topic subscribers
        self.velocity_subscriber = rospy.Subscriber(self.velocity_topic, TwistStamped, self.callback_velocity)

        # Initialize the car control publishers
        if self.control_type == "ControlRefs":
            from ros_zoe_msg import ControlRefs
            self.control_refs_publisher = rospy.Publisher(self.control_ref_topic, ControlRefs, queue_size=10) 
        else:
            self.speed_publisher = rospy.Publisher(self.speed_topic, Float32, queue_size=10)
            self.steering_angle_publisher = rospy.Publisher(self.steering_angle, Float32, queue_size=10)

        if self.navigation:
            rospy.wait_for_service(self.toogle_navigation_service_name)
            self.toogle_navigation_service = rospy.ServiceProxy(self.toogle_navigation_service_name, Empty)
        rospy.loginfo("Decision ready")


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
        if self.control_type == "ControlRefs":
            from ros_zoe_msg import ControlRefs
            ctrl = ControlRefs()
            ctrl.steerAng = steering_angle
            ctrl.linSpeed = speed
            self.control_refs_publisher.publish(ctrl)
        else:
            if speed is not None:
                self.speed_publisher.publish(speed)
            if steering_angle is not None:
                self.steering_angle_publisher.publish(steering_angle)

    def overtake(self, direction = 1):
        if not self.overtaking:
            rospy.loginfo(f"ACTION: STOP NAVIGATION")
            self.toogle_navigation()
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
    
    def hazard_lights_on(self, object):
        return object.left_blink and object.right_blink

    def toogle_navigation(self,):
        if self.navigation:
            try:
                _ = node.toogle_navigation_service()
            except rospy.ServiceException as e:
                pass
        else:
            self.publish_controls(speed=3,steering_angle=0)
            
    def publish_control_inputs(self, ):
        """Publish the control inputs to the car"""
        # STOP CAR IF OBJECT IS TOO CLOSE
        objetct_on_left_line = []
        preceding_object = None
        if not self.overtaking:
            for obj in self.object_list:
                if obj.bbox.class_id in to_not_kill:
                    obj_lane_side = self.map_handler.get_lane_side(obj)
                    if obj_lane_side == lane_side.LEFT:
                        rospy.loginfo(f"INFO: Left lane {obj.distance:.2f}m away from {self.classes[obj.bbox.class_id]}")
                        objetct_on_left_line.append(obj)
                    if obj_lane_side == lane_side.FRONT:
                        rospy.loginfo(f"INFO: Preceding {obj.distance:.2f}m away from {self.classes[obj.bbox.class_id]}")
                        preceding_object = obj
                        if (obj.distance < STOP_THRESHOLD_DISTANCE*self.real_speed or obj.distance < STOP_THRESHOLD_DISTANCE):
                            rospy.loginfo(f"ACTION: STOP (object in front is too close)")
                            self.publish_controls(speed=0, steering_angle=0)

        # START OVERTAKING FROM LEFT
        if preceding_object is not None:
            if self.real_speed < 2 and self.hazard_lights_on(preceding_object):
                if len(objetct_on_left_line) == 0: 
                    self.overtake(LEFT)
                else:
                    rospy.loginfo('INFO: object on left line => cannot overtake')
        
        # CONTINUE OVERTAKING IF 
        if self.overtaking:
            self.overtake()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <parameter-file> <publish-rate (optional)>")
    else:
        with open(sys.argv[1], "r") as parameterfile:
            config = yaml.load(parameterfile, yaml.Loader)

        rospy.init_node("decision")
        publishing_rate = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else DEFAULT_RATE
        rospy.loginfo(f"Publishing rate: {publishing_rate} Hz")
        node = Controller(config)
        rospy.loginfo(f"ACTION: START NAVIGATION")
        node.toogle_navigation()

        rate = rospy.Rate(publishing_rate)
        while not rospy.is_shutdown():
            node.publish_control_inputs()
            rate.sleep()

        rospy.spin()
