#!/usr/bin/env python3
import sys
import yaml
import rospy
import numpy as np
import cv2
import time
from cv_bridge import CvBridge

from perception.msg import ObjectList
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber

from object_detector import ObjectDetector
from distance_extractor_v2 import DistanceExtractor
from vehicle_lights import check_vehicle_lights_on


class Perception(object):
    def __init__(self, config, visualize = False, rviz_visualize = False, lidar_projection = False, log_objects = False, time_statistics = False, yolov8l = False, use_map = False):
        self.config = config
        self.visualize = visualize
        self.rviz_visualize = rviz_visualize
        self.lidar_projection = lidar_projection
        self.log_objects = log_objects
        self.time_statistics = time_statistics
        self.yolov8l = yolov8l
        self.use_map = use_map

        # Detection module
        self.distance_extractor = DistanceExtractor(config, self.lidar_projection, self.use_map)
        self.object_detector = ObjectDetector(config, yolov8l)

        self.previous_bbox_list = None
        self.previous_image = None
        self.blink_dict_history = {}

        # Publisher
        self.visualization_publisher = rospy.Publisher(config["topic"]["perception-viz"], Image, queue_size=10)
        self.object_info_publisher = rospy.Publisher(config["topic"]["object-info"], ObjectList, queue_size=10)

        # visualization utils
        if self.visualize or self.rviz_visualize:
            self.classes = None
            with open(config["model"]["detection-model-class-names-path"], 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        if self.rviz_visualize:
            self.cv_bridge = CvBridge()

        # Launch perception
        tss = ApproximateTimeSynchronizer([Subscriber(config["topic"]["forward-camera"], Image),
                            Subscriber(config["topic"]["pointcloud"], PointCloud2)], 10, 0.1, allow_headerless=True)

        tss.registerCallback(self.perception_callback)

        rospy.loginfo(f"Perception ready with parameters:")
        rospy.loginfo(f"\tVisualize: {self.visualize}")
        rospy.loginfo(f"\tLidar projection: {self.lidar_projection}")
        rospy.loginfo(f"\tLog objects: {self.log_objects}")
        rospy.loginfo(f"\tTime statistics: {self.time_statistics}")
        rospy.loginfo(f"\tYolov8l: {self.yolov8l}")
        rospy.loginfo(f"\tUse map: {self.use_map}")	

    def draw_bounding_box(self, img, class_id, x, y, x_plus_w, y_plus_h, d = None, instance_id = None, light_blink = False):
        label = ""
        if instance_id != 0:
            label = f"#{instance_id}: "
        label += str(self.classes[class_id])
        if d is not None:
            label += f": {d:.2f}m"
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if light_blink:
            cv2.putText(img, "Blink", (x - 10, y_plus_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def perception_callback(self, img_data, point_cloud_data):
        if self.time_statistics:
            overall_start = time.time()

        img = np.frombuffer(img_data.data, dtype=np.uint8).reshape((img_data.height, img_data.width, 3))

        # detect objects
        if self.time_statistics:
            start = time.time()

        bbox_list = self.object_detector.detect(img)
        current_blink_dict = {}
        th_image = None
        if bbox_list is not None and self.previous_image is not None:
            current_blink_dict, th_image = check_vehicle_lights_on(self.previous_image, self.previous_bbox_list, img, bbox_list)
            rospy.loginfo(current_blink_dict)

        self.previous_image = img.copy()
        self.previous_bbox_list = bbox_list

        if self.time_statistics:
            rospy.loginfo(f"Detection time: {time.time() - start:.2f}")

        # if objects detected -> get object position
        obj_list = []
        if len(bbox_list) > 0:
            if self.time_statistics:
                start = time.time()
            obj_list = self.distance_extractor.get_objects_position(img_data, point_cloud_data, bbox_list)
            if self.time_statistics:
                rospy.loginfo(f"Distance extraction time: {time.time() - start:.2f}")

        if (self.visualize or self.rviz_visualize) and len(obj_list) > 0:
            for obj in obj_list:
                if obj.bbox.instance_id in current_blink_dict:
                    self.draw_bounding_box(img, obj.bbox.class_id, obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.w,
                                           obj.bbox.y + obj.bbox.h, obj.distance, obj.bbox.instance_id,
                                           current_blink_dict[obj.bbox.instance_id])
        
        if self.visualize:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('perception visualization', img)
            cv2.waitKey(5)

        if self.rviz_visualize:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.visualization_publisher.publish(self.cv_bridge.cv2_to_imgmsg(img))
        
        object_list = ObjectList()
        object_list.object_list = obj_list
        self.object_info_publisher.publish(obj_list)

        if self.log_objects and len(obj_list) > 0:
            rospy.loginfo(obj_list)

        if self.time_statistics:
            rospy.loginfo(f"Overall time: {time.time() - overall_start:.2f}")

    def update_blink_history(self, current_blink):
        for instance_id in self.blink_dict_history:
            if instance_id in current_blink:
                if current_blink[instance_id]:
                    pass
                else:
                    pass
                current_blink.pop(instance_id)
            else:
                pass

        for id_instance in current_blink:
            pass




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file> [-v] [--rviz] [--lidar-projection] [--log-objects] [--time-statistics] [--yolov8l] [--use-map]]")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)

        rospy.init_node("perception")
        p = Perception(config,'-v' in sys.argv, '--rviz' in sys.argv, '--lidar-projection' in sys.argv, '--log-objects' in sys.argv, '--time-statistics' in sys.argv, '--yolov8l' in sys.argv, '--use-map' in sys.argv)
        rospy.spin()

