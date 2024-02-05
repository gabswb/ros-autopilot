import sys
import json
import yaml
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

import rospy
import tf2_ros
import transforms3d.quaternions as quaternions
from common.road import Road


class lane_side(str, Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    AWAY = "away"


class MapHandler(object):
    def __init__(self, config):
        self.config = config

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        with open(self.config["map"]["road-network-path"], 'r') as f:
            self.road_network = json.load(f)

        self.segment_dict = {}
        self.road_list = []
        self.path = []

        self.process_road_network()

    def assign_segment_to_road(self, segment, road_id):
        self.segment_dict[segment["guid"]] = road_id

    def is_segment_assigned(self, segment_guid):
        segments_assigned = list(self.segment_dict.keys())
        if segment_guid not in segments_assigned:
            return False
        else:
            return self.segment_dict[segment_guid]

    def get_segment(self, segment_guid):
        for segment in self.road_network:
            if segment["guid"] == segment_guid:
                return segment

    def get_road(self, segment):
        for road in self.road_list:
            if segment in road.segment_list:
                return road

    def process_road_network(self):
        for i in range(len(self.road_network)):
            if self.is_segment_assigned(self.road_network[i]["guid"]) is False:
                Road(self, self.road_network[i])

    def get_road_position(self, position, first_roads_to_check=None):
        if first_roads_to_check is not None:
            for road_to_check in first_roads_to_check:
                for i in range(len(road_to_check.path_to_follow) - 1):
                    x1, y1 = road_to_check.left_points[i]
                    x2, y2 = road_to_check.left_points[i + 1]
                    x3, y3 = road_to_check.right_points[i]
                    x4, y4 = road_to_check.right_points[i + 1]

                    x, y = position

                    is_in_quad = point_in_quad(x1, y1, x2, y2, x4, y4, x3, y3, x, y)

                    if is_in_quad:
                        self.path.append(road_to_check)
                        return road_to_check

        for road in self.road_list:
            for i in range(len(road.path_to_follow) - 1):
                x1, y1 = road.left_points[i]
                x2, y2 = road.left_points[i + 1]
                x3, y3 = road.right_points[i]
                x4, y4 = road.right_points[i + 1]

                x, y = position

                is_in_quad = point_in_quad(x1, y1, x2, y2, x4, y4, x3, y3, x, y)

                if is_in_quad:
                    self.path.append(road)
                    return road
        return None

    def get_lane_side(self, object):
        if np.abs(object.x) <= self.lane_width / 2 and object.z >= 0:
            return lane_side.FRONT
        elif object.x < -self.lane_width / 2 and object.x >= -self.lane_width * 3 / 2:
            return lane_side.LEFT
        elif object.x > self.lane_width / 2 and object.x <= self.lane_width * 3 / 2:
            return lane_side.RIGHT
        else:
            return lane_side.AWAY

    def get_transform(self, source_frame, target_frame):
        """Update the lidar-to-camera transform matrix from the tf topic"""
        try:
            # It’s lookup_transform(target_frame, source_frame, …) !!!
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr('Transform not found')
            return

        # Build the matrix elements
        rotation_message = transform.transform.rotation
        rotation_quaternion = np.asarray(
            (rotation_message.w, rotation_message.x, rotation_message.y, rotation_message.z))
        rotation_matrix = quaternions.quat2mat(rotation_quaternion)
        translation_message = transform.transform.translation
        translation_vector = np.asarray((translation_message.x, translation_message.y, translation_message.z)).reshape(
            3, 1)

        # Build the complete transform matrix
        return np.concatenate((
            np.concatenate((rotation_matrix, translation_vector), axis=1),
            np.asarray((0, 0, 0, 1)).reshape((1, 4))
        ), axis=0)

    def get_world_position(self, car_baslink=np.array([0, 0, 0])):
        baslink_to_map = self.get_transform("camera_forward_optical_frame", self.config["map"]["world-frame"])
        position = baslink_to_map @ np.concatenate((car_baslink, np.array([1]))).T
        return (position / position[3])[:3]

    def get_car_world_position(self, world_position):
        map_to_baslink = self.get_transform(self.config["map"]["world-frame"], "camera_forward_optical_frame")
        position = map_to_baslink @ np.concatenate((world_position, np.array([1]))).T
        return (position / position[3])[:3]


def vect_product(x1, y1, x2, y2, x3, y3):
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)


def point_in_quad(ax, ay, bx, by, cx, cy, dx, dy, px, py):
    orientation_AB_AP = vect_product(ax, ay, bx, by, px, py)
    orientation_BC_BP = vect_product(bx, by, cx, cy, px, py)
    orientation_CD_CP = vect_product(cx, cy, dx, dy, px, py)
    orientation_DA_DP = vect_product(dx, dy, ax, ay, px, py)

    if (orientation_AB_AP > 0 and orientation_BC_BP > 0 and orientation_CD_CP > 0 and orientation_DA_DP > 0) or \
            (orientation_AB_AP < 0 and orientation_BC_BP < 0 and orientation_CD_CP < 0 and orientation_DA_DP < 0):
        return True
    else:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file>")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)
        rospy.init_node("distances")
        node = MapHandler(config)
        rospy.spin()
