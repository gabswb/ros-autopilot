#!/usr/bin/env python3

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

import rospy
import tf2_ros
import transforms3d.quaternions as quaternions

from perception.msg import ObjectList
from decision.msg import DecisionInfo
from common_scripts.map_handler import MapHandler

WINDOW_SIZE_X = 75
WINDOW_SIZE_Y = 75


class MapPlotter(object):
    def __init__(self, config):
        self.config = config

        self.map_handler = MapHandler(config)
        self.road_to_check = {}

        self.objects_position = []
        self.car_position = self.map_handler.get_world_position(np.array([0, 0, 0]))

        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot([], [], 'bD' )
        self.obj_scatter = None
        self.target_scatter = None
        self.rectangle_scatter = []

        # Decision info
        self.targets = []

    def update_plot(self, frame):
        self.car_position = self.map_handler.get_world_position(np.array([0, 0, 0]))

        x = self.car_position[0]
        y = self.car_position[1]
        self.ax.set_xlim(x + WINDOW_SIZE_X // 2, x - WINDOW_SIZE_X // 2)
        self.ax.set_ylim(y + WINDOW_SIZE_Y // 2, y - WINDOW_SIZE_Y // 2)
        self.ax.set_aspect('equal')
        self.ln.set_data(x, y)

        if self.target_scatter is not None and self.target_scatter in self.ax.collections:
            self.target_scatter.remove()

        for rectangle in self.rectangle_scatter:
            rectangle.remove()
            # self.ax.add_collection(line_scatter)
        self.rectangle_scatter = []

        if self.obj_scatter is not None and self.obj_scatter in self.ax.collections:
            self.obj_scatter.remove()
        if self.objects_position:
            obj_positions = np.array(self.objects_position)
            self.obj_scatter = self.ax.scatter(obj_positions[:, 0], obj_positions[:, 1], c='r', marker='o',
                                               label='Objects')

        for road_id, available in self.road_to_check.items():
            road = self.map_handler.road_list[road_id]

            color = 'green'
            if available == 0:
                color = 'red'
            elif available == 1:
                color = 'yellow'

            polygon = Polygon(road.left_points + road.right_points[::-1], closed=True, fill=True, color=color, alpha=0.5)

            self.rectangle_scatter.append(polygon)
            self.ax.add_patch(polygon)

        if len(self.targets) > 0:
            targets = np.array(self.targets)
            self.target_scatter = self.ax.scatter(targets[:, 0], targets[:, 1], c='cornflowerblue', marker='X', label='Targets')

        return self.ln, self.obj_scatter

    def perception_callback(self, data):
        self.objects_position = []
        for obj in data.object_list:
            obj_pos = self.map_handler.get_world_position(np.array([obj.x, obj.y, obj.z]))
            self.objects_position.append(obj_pos)

    def decision_callback(self, data):
        self.targets = []
        self.road_to_check = {}
        for target in [data.target1, data.target2, data.target3, data.target4, data.target5]:
            if len(target) > 0:
                self.targets.append(target)
        for ids, available in zip(data.roads_to_check_ids, data.is_road_available):
            self.road_to_check[ids] = available

    def plot_init(self):
        for i, road in enumerate(self.map_handler.road_list):
            if road.display:
                x_left, y_left = zip(*road.left_points)
                x_middle, y_middle = zip(*road.path_to_follow)
                x_right, y_right = zip(*road.right_points)

                # Tracer le graphique
                self.ax.plot(x_left, y_left, linestyle='-', color='black')
                # if road in self.path:
                #     if road == self.path[0]:
                #         self.ax.plot(x_middle, y_middle, linestyle='-', color='red')
                #     else:
                #         self.ax.plot(x_middle, y_middle, linestyle='-', color='blue')
                self.ax.plot(x_right, y_right, linestyle='-', color='black')

        return self.ln


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <config-file>")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)
        rospy.init_node("map_plotter")
        node = MapPlotter(config)
        perception_subscriper = rospy.Subscriber(config["topic"]["object-info"], ObjectList, node.perception_callback)
        decision_subscriper = rospy.Subscriber(config["topic"]["decision-info"], DecisionInfo, node.decision_callback)
        ani = FuncAnimation(node.fig, node.update_plot, init_func=node.plot_init)
        plt.show(block=True)  # Display the initial plot
        rospy.spin()
