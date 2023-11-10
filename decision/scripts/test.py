#!/usr/bin/env python3

import sys
import math

import yaml
import numpy as np

import rospy
from std_msgs.msg import Float32, String, UInt8
from geometry_msgs.msg import TwistStamped
from perception.msg import ObjectList, Object, ObjectBoundingBox

if __name__ == "__main__":
    rospy.init_node("test")
    msg = ObjectBoundingBox()
    msg.class_id = 2
    msg.x = 4
    msg.y = 4
    msg.w = 4
    msg.h = 4
    obj = Object()
    obj.bbox = msg
    obj.x = 4
    obj.y = 4
    obj.z = 4
    obj_list = ObjectList()
    obj_list.object_list = [obj]
    empty_obj_list = ObjectList()
    empty_obj_list.object_list = []
    pub = rospy.Publisher("/perception/objects-info", ObjectList, queue_size=10)
    while input("press q to quit") != "q":
        pub.publish(obj_list)
        # pub.publish(empty_obj_list)
    # rospy.spin()
