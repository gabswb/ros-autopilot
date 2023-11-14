#!/usr/bin/env python3

import sys

import yaml
import rospy
import numpy as np
import cv2

from perception.msg import ObjectList
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber

from object_detector import ObjectDetector
from distance_extractor_v2 import DistanceExtractor 


class Perception(object):
    def __init__(self, config, visualize = False, publish = True, lidar_projection = False, log_objects = False):
        self.config = config
        self.visualize = visualize
        self.publish = publish
        self.lidar_projection = lidar_projection
        self.log_objects = log_objects

        # Detection module
        self.distance_extractor = DistanceExtractor(config)
        self.object_detector = ObjectDetector(config)

        # Publisher
        self.visualization_publisher = rospy.Publisher(config["node"]["perception-viz-topic"], Image, queue_size=10)
        self.object_info_publisher = rospy.Publisher(config["node"]["object-info-topic"], ObjectList, queue_size=10)

        # visualization utils
        if self.visualize:
            self.COLORS = np.random.uniform(0, 255, size=(80, 3))
            self.classes = None
            with open(config["model"]["detection-model-class-names-path"], 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

        # Launch perception
        tss = ApproximateTimeSynchronizer([Subscriber(config["node"]["image-topic"], Image),
                            Subscriber(config["node"]["pointcloud-topic"], PointCloud2)], 10, 0.1, allow_headerless=True)

        tss.registerCallback(self.perception_callback)
        rospy.loginfo("Perception ready")	

    def draw_bounding_box(self, img, class_id, x, y, x_plus_w, y_plus_h, d = None):
        label = str(self.classes[class_id])
        if d is not None:
            label += f": {d:.2f}m"
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def perception_callback(self, img_data, point_cloud_data):
        
        img = np.frombuffer(img_data.data, dtype=np.uint8).reshape((img_data.height, img_data.width, 3))
        
        # detect objects
        bbox_list = self.object_detector.detect(img)

        # if objects detected -> get object position
        obj_list = []
        if len(bbox_list) > 0:
            obj_list = self.distance_extractor.get_objects_position(img_data, point_cloud_data, bbox_list)
        
        if self.visualize and len(obj_list) > 0:
            for obj in obj_list:
                self.draw_bounding_box(img, obj.bbox.class_id, obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.w, obj.bbox.y + obj.bbox.h, obj.distance)
        
        if self.visualize:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('perception visualization', img)
            cv2.waitKey(5)
            
        if self.lidar_projection:
            self.distance_extractor.project_lidar_to_image(img_data, point_cloud_data)
        
        if self.publish and len(obj_list) > 0:
            object_list = ObjectList()
            object_list.object_list = obj_list
            self.object_info_publisher.publish(obj_list)

        if self.log_objects and len(obj_list) > 0:
            rospy.loginfo(obj_list)
        


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file> [-v] [--no-publish] [--lidar-projection] [--log-objects]]")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)

        rospy.init_node("perception")
        p = Perception(config, '-v' in sys.argv, '--no-publish' not in sys.argv, '--lidar-projection' in sys.argv, '--log-objects')
        rospy.spin()

