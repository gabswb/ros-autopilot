#!/usr/bin/env python3

import sys
import yaml

from sensor_msgs.msg import Image, PointCloud2

from cv_bridge import CvBridge
from object_detector import ObjectDetector
from message_filters import ApproximateTimeSynchronizer, Subscriber
from distance_extractor import DistanceExtractor 
import rospy
import cv2
import numpy as np

# printing utils
cvbridge = CvBridge()
COLORS = np.random.uniform(0, 255, size=(80, 3))
classes = None

def draw_bounding_box(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
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

        def perception_callback(img_data, point_cloud_data):
            rospy.loginfo(f"{point_cloud_data.header.stamp} = {img_data.header.stamp}")
            rospy.loginfo(f"{point_cloud_data.header.stamp == img_data.header.stamp}")


        
            # img = np.frombuffer(img_data.data, dtype=np.uint8).reshape((img_data.height, img_data.width, 3))
            # # detect objects
            # obj_bbox_list = object_detector.detect(img)

            # # if objects detected, compute distance
            # if len(obj_bbox_list) > 0:
            #     distance_extractor.compute_distance(img, img_data, point_cloud_data)

            # if visualize:
            #     for obj_bbox in obj_bbox_list:
            #         draw_bounding_box(img, obj_bbox.class_id, round(obj_bbox.x), round(obj_bbox.y), round(obj_bbox.x + obj_bbox.w), round(obj_bbox.y + obj_bbox.h))
            
            # if visualize:
            #     visualization_publisher.publish(cvbridge.cv2_to_imgmsg(img, encoding="passthrough"))
        
        rospy.loginfo("Perception ready")	

        tss = ApproximateTimeSynchronizer([Subscriber(config["node"]["image-topic"], Image),
                            Subscriber(config["node"]["pointcloud-topic"], PointCloud2)], 10, 0.1, allow_headerless=True)

        tss.registerCallback(perception_callback)
        rospy.spin()

