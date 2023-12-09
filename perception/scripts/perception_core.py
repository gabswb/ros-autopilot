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
from distance_extractor import DistanceExtractor
from vehicle_lights import VehicleLightsDetector, BLINK_HEIGHT, RIGHT_BLINK, LEFT_BLINK, HAS_LIGHT

DISPLAY_LIGHT_BBOX = True


class Perception(object):
    def __init__(self, config, rviz_visualize = False, lidar_projection = False, log_objects = False, time_statistics = False, yolov8l = False, use_map = False, detect_lights = True, backward_camera = True):
        self.config = config
        self.rviz_visualize = rviz_visualize
        self.lidar_projection = lidar_projection
        self.log_objects = log_objects
        self.time_statistics = time_statistics
        self.yolov8l = yolov8l
        self.use_map = use_map
        self.detect_lights = detect_lights
        self.backward_camera = backward_camera

        # Detection module
        self.forward_distance_extractor = DistanceExtractor(config, self.config["topic"]["forward-camera-info"], self.config["topic"]["forward-lidar-viz"], self.lidar_projection, self.use_map)
        self.backward_distance_extractor = DistanceExtractor(config, self.config["topic"]["backward-camera-info"], self.config["topic"]["backward-lidar-viz"], self.lidar_projection, self.use_map)
        self.vehicle_light_detector = VehicleLightsDetector()
        self.object_detector = ObjectDetector(config, yolov8l)
        self.surrounding_object_detector = ObjectDetector(config, yolov8l, False)

        # Publisher
        self.forward_bbox_publisher = rospy.Publisher(config["topic"]["forward-bbox-viz"], Image, queue_size=10)
        self.backward_bbox_publisher = rospy.Publisher(config["topic"]["backward-bbox-viz"], Image, queue_size=10)
        self.object_info_publisher = rospy.Publisher(config["topic"]["object-info"], ObjectList, queue_size=10)

        # visualization utils
        if self.rviz_visualize:
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
        rospy.loginfo(f"\tRViz visualize: {self.rviz_visualize}")
        rospy.loginfo(f"\tLidar projection: {self.lidar_projection}")
        rospy.loginfo(f"\tLog objects: {self.log_objects}")
        rospy.loginfo(f"\tTime statistics: {self.time_statistics}")
        rospy.loginfo(f"\tYolov8l: {self.yolov8l}")
        rospy.loginfo(f"\tUse map: {self.use_map}")	
        rospy.loginfo(f"\tDetect light: {self.detect_lights}")

    def perception_callback(self, forward_img_data, forward_left_img_data, forward_right_img_data, backward_img_data, pointcloud_data):
        if self.time_statistics:
            overall_start = time.time()

        if self.backward_camera:
            perception_pipeline = [(forward_img_data, self.forward_distance_extractor, self.forward_bbox_publisher),
                                   (backward_img_data, self.backward_distance_extractor, self.backward_bbox_publisher)]
        else:
            perception_pipeline = [(forward_img_data, self.forward_distance_extractor, self.forward_bbox_publisher)]
        
        object_list = []

        for image_data, distance_extractor, publisher in perception_pipeline:
            image = np.frombuffer(image_data.data, dtype=np.uint8).reshape((image_data.height, image_data.width, 3))

            if self.time_statistics:
                start = time.time()
            bbox_list = self.object_detector.detect(image)
            if self.time_statistics:
                rospy.loginfo(f"Detection time: {time.time() - start:.2f}")
            
            if self.time_statistics:
                start = time.time()
            objects = distance_extractor.get_objects_position(image_data, pointcloud_data, bbox_list)
            if self.time_statistics:
                rospy.loginfo(f"Distance extraction time: {time.time() - start:.2f}")

            if self.time_statistics:
                start = time.time()
            objects = self.vehicle_light_detector.check_lights(image, objects)
            if self.time_statistics:
                rospy.loginfo(f"Light detection time: {time.time() - start:.2f}")   

            if self.rviz_visualize:
                for obj in objects:
                    self.draw_bounding_box(image, obj.bbox.class_id, obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.w,
                                        obj.bbox.y + obj.bbox.h, obj.distance, obj.bbox.instance_id, obj.left_blink, obj.right_blink)
                
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                publisher.publish(self.cv_bridge.cv2_to_imgmsg(image))
                
            object_list.extend(objects)
        
        object_list_msg = ObjectList()
        object_list_msg.object_list = object_list
        self.object_info_publisher.publish(object_list_msg)

        if self.log_objects and len(object_list) > 0:
            rospy.loginfo(object_list_msg)

        if self.time_statistics:
            rospy.loginfo(f"Overall time: {time.time() - overall_start:.2f}")



    def draw_bounding_box(self, img, class_id, x, y, x_plus_w, y_plus_h, d = None, instance_id = None, left_blink = False, right_blink = False):
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
        
        if left_blink:
            cv2.putText(img, "L Blink", (x - 10, y_plus_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if right_blink:
            cv2.putText(img, "R Blink", (x_plus_w - 10, y_plus_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file> [--rviz] [--lidar-projection] [--log-objects] [--time-statistics] [--yolov8l] [--use-map] [--no-lights] [--backward-camera]]")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)

        rospy.init_node("perception")
        p = Perception(config,
                        '--rviz' in sys.argv,
                        '--lidar-projection' in sys.argv,
                        '--log-objects' in sys.argv,
                        '--time-statistics' in sys.argv,
                        '--yolov8l' in sys.argv,
                        '--use-map' in sys.argv,
                        not '--no-lights' in sys.argv,
                        '--backward-camera' in sys.argv)
        rospy.spin()

