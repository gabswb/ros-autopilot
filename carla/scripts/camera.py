from sensor import Sensor
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import cv2 as cv
import rospy

import math


class Camera(Sensor):
    def __init__(self, carla_object, frame_id, child_frame_id, topic_name, camera_info_topic_name, position):
        self.camera_publisher = rospy.Publisher(topic_name, Image, queue_size=10)
        self.camera_info_publisher = rospy.Publisher(camera_info_topic_name, CameraInfo,
                                                               queue_size=10)

        self.cv_bridge = CvBridge()

        super().__init__(carla_object, frame_id, child_frame_id, topic_name, position)

        self.camera_info = self.get_camera_info(carla_object)

    def sensor_callback(self, image):
        # print(image.frame_number)
        i = np.array(image.raw_data)
        i2 = i.reshape(image.height, image.width, 4)
        i3 = i2[:, :, :3]
        bgr_img = cv.cvtColor(i3, cv.COLOR_RGB2BGR)

        # self.carla_fwd_camera_info_publisher.publish(self.camera_info)
        img_msg = self.cv_bridge.cv2_to_imgmsg(bgr_img)
        img_msg.header = self.get_msg_header()
        self.camera_publisher.publish(img_msg)

    def get_camera_info(self, camera):
        camera_info = CameraInfo()
        # store info without header
        camera_info.header = self.get_msg_header()
        camera_info.width = int(camera.attributes['image_size_x'])
        camera_info.height = int(camera.attributes['image_size_y'])
        camera_info.distortion_model = 'mei'
        cx = camera_info.width / 2.0
        cy = camera_info.height / 2.0
        fx = camera_info.width / (
                2.0 * math.tan(float(camera.attributes['fov']) * math.pi / 360.0))
        fy = fx
        camera_info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        camera_info.D = [0.0]
        camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        return camera_info
