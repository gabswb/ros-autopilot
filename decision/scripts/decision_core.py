#!/usr/bin/env python3

import sys
import yaml
import numpy as np

import rospy
from std_msgs.msg import Float32, String, UInt8
from std_srvs.srv import Empty
from geometry_msgs.msg import TwistStamped

from perception.msg import ObjectList, Object, ObjectBoundingBox
from decision.msg import CameraActivation

from common_scripts.map_handler import MapHandler, lane_side
from decision_state import DECISION_STATE
from controller import Controller

to_not_kill = (0,1,2,3,4,5,6,7,8, 25) #person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, umbrella
STOP_THRESHOLD_DISTANCE = 10 # meters
DEFAULT_RATE = 100 # Hz

LANE_WIDTH = 1.5
LEFT = -1
RIGHT = 1



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
        self.control_ref_topic = self.config['topic']['control-refs-topic']
        self.camera_activation_topic = self.config['topic']['camera-activation']

        # Situation states
        self.real_speed = None
        self.real_angular_speed = None
        self.object_list = []
        self.target_position = None

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

        # Controller 
        self.controller = Controller(config)
        self.camera_activation = rospy.Publisher(self.camera_activation_topic,CameraActivation,queue_size=10)

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
    
    def compute_path_for_overtaking(self, ):
        # FIXME use kalman filter to predict/match position of overtaken car
        # then compute path to overtake
        target_position = [self.overtaken_car.x-LANE_WIDTH*1.5, self.overtaken_car.z-1]
        return target_position
    
    def activate_camera(self, forward=True,forward_left=False,forward_right=False,backward=True):
        cam_act = CameraActivation()
        cam_act.forward = forward
        cam_act.forward_left = forward_left
        cam_act.forward_right = forward_right
        cam_act.backward = backward
        self.camera_activation.publish(cam_act) 
            
    def decision_maker(self, ):
        # STOP CAR IF OBJECT IS TOO CLOSE
        objetct_on_left_line = []
        objetct_on_right_line = []
        current_state =  []
        target_position = None
        preceding_object = None
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
                            current_state.append(DECISION_STATE.STOP)
                        elif obj.distance < STOP_THRESHOLD_DISTANCE*self.real_speed:
                            current_state.append(DECISION_STATE.ADAPT_TO_FORWARD_SPEED)


        # START OVERTAKING FROM LEFT
        if not self.overtaking and preceding_object is not None:
            if self.real_speed < 1 and self.hazard_lights_on(preceding_object):
                if len(objetct_on_left_line) == 0: 
                    self.overtaking = True
                    self.overtaken_car = preceding_object
                    current_state.append(DECISION_STATE.STOP_NAVIGATION)
                    self.activate_camera(forward_right=True)
                    self.target_position = self.compute_path_for_overtaking()
                    print(self.target_position)
                else:
                    rospy.loginfo('INFO: object on left line => cannot overtake')
        
        # CONTINUE OVERTAKING IF STARTED
        if self.overtaking == True:
            current_state.append(DECISION_STATE.OVERTAKE)

        self.controller.handle_decision(current_state, self.target_position, self.real_speed)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <parameter-file> <publish-rate (optional)>")
    else:
        with open(sys.argv[1], "r") as parameterfile:
            config = yaml.load(parameterfile, yaml.Loader)

        rospy.init_node("decision")
        publishing_rate = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else DEFAULT_RATE
        rospy.loginfo(f"Publishing rate: {publishing_rate} Hz")
        node = DecisionMaker(config, publishing_rate)
        node.controller.handle_decision([DECISION_STATE.START_NAVIGATION])

        rate = rospy.Rate(publishing_rate)
        while not rospy.is_shutdown():
            node.decision_maker()
            rate.sleep()

        rospy.spin()
