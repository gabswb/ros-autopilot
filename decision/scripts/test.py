#!/usr/bin/env python2.7

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
    obj = Object()
    obj_bbx = ObjectBoundingBox()
    obj_bbx.x = 0
    obj_bbx.y = 0
    obj_bbx.w = 0
    obj_bbx.h = 0
    obj.bbox = obj_bbx
    obj.distance = 1.0
    obj.x = 0.0
    obj.y = 0.0
    obj.z = 0.0
    obj_list = ObjectList()
    obj_list.object_list = [obj]
    empty_obj_list = ObjectList()
    empty_obj_list.object_list = []
    pub = rospy.Publisher("/perception/objects-info", ObjectList, queue_size=10)
    speed_publisher = rospy.Publisher("/ZOE2UTBM/control/speed", Float32, queue_size=10)
    v = input("press q to quit")
    while v != "q":
        speed_publisher.publish(float(v))
        v = input("press q to quit")
        # pub.publish(empty_obj_list)
    # rospy.spin()
