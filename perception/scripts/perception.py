#!/usr/bin/env python3

import sys
import yaml

from sensor_msgs.msg import Image, PointCloud2

from object_detector import ObjectDetector
from message_filters import ApproximateTimeSynchronizer, Subscriber
from distance_extractor import DistanceExtractor 
import rospy
import cv2
import numpy as np

# from perception.msg import ObjectList

# printing utils
COLORS = np.random.uniform(0, 255, size=(80, 3))
classes = None

def draw_bounding_box(img, class_id, x, y, x_plus_w, y_plus_h, d = None):
    label = str(classes[class_id])
    if d is not None:
        label += f": {d:.2f}m"
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file>")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)
        rospy.init_node("perception")

        visualize = True
        with open(config["model"]["detection-model-class-names-path"], 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        distance_extractor = DistanceExtractor(config)
        object_detector = ObjectDetector(config)
        visualization_publisher = rospy.Publisher(config["node"]["perception-viz-topic"], Image, queue_size=10)
        object_info_publisher = rospy.Publisher(config["node"]["object-info-topic"], Image, queue_size=10)

        def perception_callback(img_data, point_cloud_data):
            img = np.frombuffer(img_data.data, dtype=np.uint8).reshape((img_data.height, img_data.width, 3))
            
            # detect objects
            obj_list = object_detector.detect(img)

            # if objects detected -> get object position
            if len(obj_list) > 0:
                obj_list = distance_extractor.get_object_positions(img, img_data, point_cloud_data, obj_list)
            

            if visualize and len(obj_list) > 0:
                for obj in obj_list:
                    draw_bounding_box(img, obj.bbox.class_id, round(obj.bbox.x), round(obj.bbox.y), round(obj.bbox.x + obj.bbox.w), round(obj.bbox.y + obj.bbox.h), np.linalg.norm([obj.x, obj.y, obj.z]))
                
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow('dist', img)
                cv2.waitKey(5) 

            # publish object list
            # object_list = ObjectList()
            # for obj in obj_list:
                # obj.to_msg()
            # object_list.object_list = obj_list
            # object_info_publisher.publish(object_list)
        
        rospy.loginfo("Perception ready")	

        tss = ApproximateTimeSynchronizer([Subscriber(config["node"]["image-topic"], Image),
                            Subscriber(config["node"]["pointcloud-topic"], PointCloud2)], 10, 0.1, allow_headerless=True)

        tss.registerCallback(perception_callback)
        rospy.spin()

