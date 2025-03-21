#!/usr/bin/env python3
import sys
import yaml
import rospy
import numpy as np
import cv2
import time
from cv_bridge import CvBridge

from perception.msg import ObjectList
from decision.msg import CameraActivation
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber

from object_detector import ObjectDetector
from distance_extractor import DistanceExtractor
from common.map_handler import MapHandler
from vehicle_lights import VehicleLightsDetector, BLINK_HEIGHT, RIGHT_BLINK, LEFT_BLINK, HAS_LIGHT

DISPLAY_LIGHT_BBOX = True

usage_msg = f"\
Usage: {__file__} <config-file> [options]\n\
    -h, --help      print this message\n\
    --rviz          publish perception visualisation\n\
    --verbose       verbose mode, print detected objects\n\
    --yolov8l       use yolov8l model for better accuracy\n\
    --use-map       use structural map to filter out object not on the road"

class Perception(object):
    def __init__(self, config, rviz_visualize = False, lidar_projection = False, log_objects = False, time_statistics = False, yolov8l = False, use_map = False, detect_lights = True, only_front_camera = True):
        self.config = config
        self.rviz_visualize = rviz_visualize
        self.lidar_projection = lidar_projection
        self.log_objects = log_objects
        self.time_statistics = time_statistics
        self.yolov8l = yolov8l
        self.use_map = use_map
        self.detect_lights = detect_lights
        self.only_front_camera = only_front_camera

        # Detection module
        self.forward_distance_extractor = DistanceExtractor(config, self.config["topic"]["forward-camera-info"], self.config["topic"]["forward-lidar-viz"], self.lidar_projection, self.use_map)
        if not self.only_front_camera:
            self.backward_distance_extractor = DistanceExtractor(config, self.config["topic"]["backward-camera-info"], self.config["topic"]["backward-lidar-viz"], self.lidar_projection, self.use_map)
            self.surrounding_object_detector = ObjectDetector(config, yolov8l, False)
        self.vehicle_light_detector = VehicleLightsDetector()
        self.object_detector = ObjectDetector(config, yolov8l)
        self.map_handler = MapHandler(config)

        # Camera activation states
        self.camera_activation_topic = self.config['topic']['camera-activation']
        rospy.Subscriber(self.camera_activation_topic, CameraActivation, self.callback_camera_activation)
        self.forward_camera = True
        self.forward_left_camera = False
        self.forward_right_camera = False
        self.backward_camera = True

        # Publisher
        self.forward_bbox_publisher = rospy.Publisher(config["topic"]["forward-bbox-viz"], Image, queue_size=10)
        if not self.only_front_camera:
            self.backward_bbox_publisher = rospy.Publisher(config["topic"]["backward-bbox-viz"], Image, queue_size=10)
        self.object_info_publisher = rospy.Publisher(config["topic"]["object-info"], ObjectList, queue_size=10)

        # Visualization utils
        if self.rviz_visualize:
            self.classes = None
            with open(config["model"]["detection-model-class-names-path"], 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        if self.rviz_visualize:
            self.cv_bridge = CvBridge()

        # Launch perception
        if self.only_front_camera:
            tss = ApproximateTimeSynchronizer([Subscriber(config["topic"]["forward-camera"], Image),
                                               Subscriber(config["topic"]["pointcloud"], PointCloud2)], 10, 0.1, allow_headerless=True)
            tss.registerCallback(self.singlecamera_perception_callback)
        else:
            tss = ApproximateTimeSynchronizer([Subscriber(config["topic"]["forward-camera"], Image),
                                           Subscriber(config["topic"]["forward-left-camera"], Image),
                                           Subscriber(config["topic"]["forward-right-camera"], Image),
                                           Subscriber(config["topic"]["backward-camera"], Image),
                            Subscriber(config["topic"]["pointcloud"], PointCloud2)], 10, 0.1, allow_headerless=True)
            tss.registerCallback(self.multicamera_perception_callback)


        rospy.loginfo(f"Perception ready with parameters:")
        rospy.loginfo(f"\tRViz visualize: {self.rviz_visualize}")
        rospy.loginfo(f"\tLidar projection: {self.lidar_projection}")
        rospy.loginfo(f"\tLog objects: {self.log_objects}")
        rospy.loginfo(f"\tTime statistics: {self.time_statistics}")
        rospy.loginfo(f"\tYolov8l: {self.yolov8l}")
        rospy.loginfo(f"\tUse map: {self.use_map}")	
        rospy.loginfo(f"\tDetect light: {self.detect_lights}")
        rospy.loginfo(f"\tOnly front camera: {self.only_front_camera}")

    def callback_camera_activation(self, data):
        """Callback for camera activation topic"""
        self.backward_camera = data.backward
        self.forward_left_camera = data.forward_left
        self.forward_right_camera = data.forward_right
        self.forward_camera = data.forward
        rospy.loginfo(f'Camera activation: forward:{self.forward_camera} forward_left:{self.forward_left_camera} forward_right:{self.forward_right_camera} backward:{self.backward_camera}')

    def multicamera_perception_callback(self, forward_img_data, forward_left_img_data, forward_right_img_data, backward_img_data, pointcloud_data):
        """Callback for perception topic"""
        if self.time_statistics:
            overall_start = time.time()

        perception_pipeline = []
        if self.forward_camera:
            perception_pipeline.append((forward_img_data, self.object_detector, self.forward_distance_extractor, self.forward_bbox_publisher))
        for is_activated, img_data in zip([self.forward_left_camera, self.forward_right_camera],[forward_left_img_data, forward_right_img_data]):
            if is_activated:
                perception_pipeline.append((img_data, self.surrounding_object_detector, self.forward_distance_extractor, self.forward_bbox_publisher))
        if self.backward_camera:
            perception_pipeline.append((backward_img_data, self.surrounding_object_detector, self.backward_distance_extractor, self.backward_bbox_publisher))
        
        object_list = []

        for image_data, object_detector, distance_extractor, publisher in perception_pipeline:
            image = np.frombuffer(image_data.data, dtype=np.uint8).reshape((image_data.height, image_data.width, 3))

            if self.time_statistics:
                start = time.time()
            bbox_list = object_detector.detect(image)
            if self.time_statistics:
                rospy.loginfo(f"Detection time: {time.time() - start:.2f}")
            
            if self.time_statistics:
                start = time.time()
            objects = distance_extractor.get_objects_position(image_data, pointcloud_data, bbox_list)
            if self.time_statistics:
                rospy.loginfo(f"Distance extraction time: {time.time() - start:.2f}")

            if self.detect_lights and image_data.header.frame_id == "camera_forward_optical_frame":
                if self.time_statistics:
                    start = time.time()
                objects = self.vehicle_light_detector.check_lights(image, objects)
                if self.time_statistics:
                    rospy.loginfo(f"Light detection time: {time.time() - start:.2f}")   

            if self.rviz_visualize and image_data.header.frame_id == "camera_forward_optical_frame":
                for obj in objects:
                    self.draw_bounding_box(image, obj.bbox.class_id, obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.w,
                                        obj.bbox.y + obj.bbox.h, obj, obj.distance, obj.bbox.instance_id, obj.left_blink, obj.right_blink)
                
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

    def singlecamera_perception_callback(self, image_data, pointcloud_data):
        """Callback for perception topic"""
        if self.time_statistics:
            overall_start = time.time()
        
        object_list = []

        image = np.frombuffer(image_data.data, dtype=np.uint8).reshape((image_data.height, image_data.width, 3))

        if self.time_statistics:
            start = time.time()
        bbox_list = self.object_detector.detect(image)
        if self.time_statistics:
            rospy.loginfo(f"Detection time: {time.time() - start:.2f}")
        
        if self.time_statistics:
            start = time.time()
        objects = self.forward_distance_extractor.get_objects_position(image_data, pointcloud_data, bbox_list)
        if self.time_statistics:
            rospy.loginfo(f"Distance extraction time: {time.time() - start:.2f}")    
        
        if self.detect_lights:
            if self.time_statistics:
                start = time.time()
            objects = self.vehicle_light_detector.check_lights(image, objects)
            if self.time_statistics:
                rospy.loginfo(f"Light detection time: {time.time() - start:.2f}")   

        if self.rviz_visualize:
            for obj in objects:
                self.draw_bounding_box(image, obj.bbox.class_id, obj.bbox.x, obj.bbox.y, obj.bbox.x + obj.bbox.w,
                                    obj.bbox.y + obj.bbox.h, obj, obj.distance, obj.bbox.instance_id, obj.left_blink, obj.right_blink)
            
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.forward_bbox_publisher.publish(self.cv_bridge.cv2_to_imgmsg(image))
                
        object_list.extend(objects)
        
        object_list_msg = ObjectList()
        object_list_msg.object_list = object_list
        self.object_info_publisher.publish(object_list_msg)

        if self.log_objects and len(object_list) > 0:
            rospy.loginfo(object_list_msg)

        if self.time_statistics:
            rospy.loginfo(f"Overall time: {time.time() - overall_start:.2f}")

    def draw_bounding_box(self, img, class_id, x, y, x_plus_w, y_plus_h, object, d = None, instance_id = None, left_blink = False, right_blink = False):
        """Draw bounding box on the image"""
        label = ""
        if instance_id != 0:
            label = f"#{instance_id}: "
        label += str(self.classes[class_id])
        if d is not None:
            label += f": {d:.2f}m"
        
        # draw bouding box
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        
        # draw blinkers
        if left_blink or right_blink:
            l_x = x_plus_w - x
            l_y = y_plus_h - y

            y_min = y + int(l_y * BLINK_HEIGHT[0])
            y_max = y + int(l_y * BLINK_HEIGHT[1])

            if left_blink:
                x_min_left = x + int(l_x * LEFT_BLINK[0])
                x_max_left = x + int(l_x * LEFT_BLINK[1])
                cv2.rectangle(img, (x_min_left,y_min), (x_max_left,y_max), (255, 0, 0), 1)
            
            if right_blink:
                x_min_right = x + int(l_x * RIGHT_BLINK[0])
                x_max_right = x + int(l_x * RIGHT_BLINK[1])
                cv2.rectangle(img, (x_min_right,y_min), (x_max_right,y_max), (255, 0, 0), 1)

        # print label and instance id
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        


if __name__ == "__main__":
    if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
        print(usage_msg)
        #print(f"Usage : {sys.argv[0]} <config-file> [--rviz] [--lidar-projection] [--log-objects] [--time-statistics] [--yolov8l] [--use-map] [--no-lights] [--only-front-camera]]")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)

        rospy.init_node("perception")
        p = Perception(config,
                        '--rviz' in sys.argv,
                        '--lidar-projection' in sys.argv,
                        '--verbose' in sys.argv or '-v' in sys.argv,
                        '--time-statistics' in sys.argv,
                        '--yolov8l' in sys.argv,
                        '--use-map' in sys.argv,
                        not '--no-lights' in sys.argv,
                        '--only-front-camera' in sys.argv)
        rospy.spin()

