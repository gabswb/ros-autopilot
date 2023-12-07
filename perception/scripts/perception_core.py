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
from vehicle_lights import VehicleLightsDetector, BLINK_HEIGHT, RIGHT_BLINK, LEFT_BLINK, HAS_LIGHT

DISPLAY_LIGHT_BBOX = True


class Perception(object):
    def __init__(self, config, visualize = False, rviz_visualize = False, lidar_projection = False, log_objects = False, time_statistics = False, yolov8l = False, use_map = False, detect_lights = True):
        self.config = config
        self.visualize = visualize
        self.rviz_visualize = rviz_visualize
        self.lidar_projection = lidar_projection
        self.log_objects = log_objects
        self.time_statistics = time_statistics
        self.yolov8l = yolov8l
        self.use_map = use_map
        self.detect_lights = detect_lights

        # Detection module
        self.distance_extractor = DistanceExtractor(config, self.lidar_projection, self.use_map)
        self.object_detector = ObjectDetector(config, yolov8l)
        self.vehicle_light_detector = VehicleLightsDetector()
        self.surrounding_object_detector = ObjectDetector(config, yolov8l, False)

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
                                           Subscriber(config["topic"]["forward-left-camera"], Image),
                                           Subscriber(config["topic"]["forward-right-camera"], Image),
                                           Subscriber(config["topic"]["backward-camera"], Image),
                            Subscriber(config["topic"]["pointcloud"], PointCloud2)], 10, 0.1, allow_headerless=True)

        tss.registerCallback(self.perception_callback)

        rospy.loginfo(f"Perception ready with parameters:")
        rospy.loginfo(f"\tOpencv visualize: {self.visualize}")
        rospy.loginfo(f"\tRViz visualize: {self.rviz_visualize}")
        rospy.loginfo(f"\tLidar projection: {self.lidar_projection}")
        rospy.loginfo(f"\tLog objects: {self.log_objects}")
        rospy.loginfo(f"\tTime statistics: {self.time_statistics}")
        rospy.loginfo(f"\tYolov8l: {self.yolov8l}")
        rospy.loginfo(f"\tUse map: {self.use_map}")	
        rospy.loginfo(f"\tDetect light: {self.detect_lights}")

    def draw_bounding_box(self, img, class_id, x, y, x_plus_w, y_plus_h, d = None, instance_id = None, left_blink = 0, right_blink = 0):
        label = ""
        if instance_id != 0:
            label = f"#{instance_id}: "
        label += str(self.classes[class_id])
        if d is not None:
            label += f": {d:.2f}m"
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        if class_id in HAS_LIGHT and DISPLAY_LIGHT_BBOX and self.detect_lights:
            l_x = x_plus_w - x
            l_y = y_plus_h - y

            y_min = y + int(l_y * BLINK_HEIGHT[0])
            y_max = y + int(l_y * BLINK_HEIGHT[1])

            x_min_left = x + int(l_x * LEFT_BLINK[0])
            x_max_left = x + int(l_x * LEFT_BLINK[1])

            x_min_right = x + int(l_x * RIGHT_BLINK[0])
            x_max_right = x + int(l_x * RIGHT_BLINK[1])

            cv2.rectangle(img, (x_min_left,y_min), (x_max_left,y_max), (255, 0, 0), 1)
            cv2.rectangle(img, (x_min_right,y_min), (x_max_right,y_max), (255, 0, 0), 1)

        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if left_blink > 0:
            cv2.putText(img, "L Blink", (x - 10, y_plus_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if right_blink > 0:
            cv2.putText(img, "R Blink", (x_plus_w - 10, y_plus_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def perception_callback(self, forward_img_data, forward_left_img_data, forward_right_img_data, backward_img_data, point_cloud_data):
        if self.time_statistics:
            overall_start = time.time()

        forward_img = np.frombuffer(forward_img_data.data, dtype=np.uint8).reshape((forward_img_data.height, forward_img_data.width, 3))

        # detect objects
        if self.time_statistics:
            start = time.time()
        bbox_list = self.object_detector.detect(forward_img)
        if self.time_statistics:
            rospy.loginfo(f"Detection time: {time.time() - start:.2f}")

        # if objects detected -> get object position
        obj_list = []
        if len(bbox_list) > 0:
            if self.time_statistics:
                start = time.time()
            obj_list = self.distance_extractor.get_objects_position(forward_img_data, point_cloud_data, bbox_list)
            if self.time_statistics:
                rospy.loginfo(f"Distance extraction time: {time.time() - start:.2f}")

        # check vehicle lights
        if self.detect_lights:
            if self.time_statistics:
                start = time.time()
            obj_list = self.vehicle_light_detector.check_lights(forward_img, obj_list)
            if self.time_statistics:
                rospy.loginfo(f"Light detection time: {time.time() - start:.2f}")        

        if (self.visualize or self.rviz_visualize) and len(obj_list) > 0:
            for obj in obj_list:
                self.draw_bounding_box(forward_img, obj.bbox.class_id, obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.w,
                                        obj.bbox.y + obj.bbox.h, obj.distance, obj.bbox.instance_id, obj.left_blink, obj.right_blink)
        
        if self.visualize:
            forward_img = cv2.cvtColor(forward_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('perception visualization', forward_img)
            cv2.waitKey(5)

        if self.rviz_visualize:
            forward_img = cv2.cvtColor(forward_img, cv2.COLOR_RGB2BGR)
            self.visualization_publisher.publish(self.cv_bridge.cv2_to_imgmsg(forward_img))
        
        object_list = ObjectList()
        object_list.object_list = obj_list
        self.object_info_publisher.publish(obj_list)

        if self.log_objects and len(obj_list) > 0:
            rospy.loginfo(obj_list)

        if self.time_statistics:
            rospy.loginfo(f"Overall time: {time.time() - overall_start:.2f}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file> [-v] [--rviz] [--lidar-projection] [--log-objects] [--time-statistics] [--yolov8l] [--use-map] [--no-lights]")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)

        rospy.init_node("perception")
        p = Perception(config,'-v' in sys.argv, '--rviz' in sys.argv, '--lidar-projection' in sys.argv, '--log-objects' in sys.argv, '--time-statistics' in sys.argv, '--yolov8l' in sys.argv, '--use-map' in sys.argv, not '--no-lights' in sys.argv)
        rospy.spin()

