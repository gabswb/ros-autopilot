import sys
import json
import yaml
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import itertools

colors = ["g", "r", "orange", "y", "black", "m", "cyan", "magenta"]


class lane_side(str, Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    AWAY = "away"


class Road():
    id_obj = itertools.count()

    def __init__(self, map_handler, first_segment):
        self.id = next(Road.id_obj)
        self.segment_list = []

        self.left_points = []
        self.right_points = []

        self.oppositeRoad = None
        self.leftRoad = None
        self.forwardRoad = None
        self.rightRoad = None

        self.display = True

        self.road_network = map_handler.road_network
        self.map_handler = map_handler

        self.find_other_segments(first_segment)
        print(self.segment_list)

    def find_other_segments(self, first_segment):
        # segments_id = [segment["guid"] for segment in self.road_network]

        self.add_segment_to_road(first_segment, "next")

        next_segment = first_segment
        previous_segment = first_segment

        while len(previous_segment["previous"]) == 1:
            segment_road = self.map_handler.is_segment_assigned(previous_segment["previous"][0])
            if segment_road is False:
                previous_segment_guid = previous_segment["previous"][0]
                previous_segment = self.map_handler.get_segment(previous_segment_guid)
                self.add_segment_to_road(previous_segment, "previous")
            else:
                break

        while len(next_segment["next"]) == 1:
            segment_road = self.map_handler.is_segment_assigned(next_segment["next"][0])
            if segment_road is False:
                next_segment_guid = next_segment["next"][0]
                next_segment = self.map_handler.get_segment(next_segment_guid)
                self.add_segment_to_road(next_segment, "next")
            else:
                break

        if len(next_segment["next"]) >= 2:
            last_index = int(len(next_segment["geometry"]) - 1)
            origin_point = [next_segment["geometry"][last_index]['px'],
                            next_segment["geometry"][last_index]['py']]

            normal_point = [next_segment["geometry"][last_index]['px'] +
                            next_segment["geometry"][last_index]['tx'],
                            next_segment["geometry"][last_index]['py'] +
                            next_segment["geometry"][last_index]['ty']]

            intersection_segment_list = []
            angle_list = []
            point_list = []
            for i in range(len(next_segment["next"])):
                intersection_segment_guid = next_segment["next"][i]

                intersection_segment = self.map_handler.get_segment(intersection_segment_guid)
                intersection_segment_list.append(intersection_segment)

                middle_index = int(len(intersection_segment["geometry"]) / 2)
                point = [intersection_segment["geometry"][middle_index]['px'],
                         intersection_segment["geometry"][middle_index]['py']]

                point_list.append(point)
                angle = get_angle(normal_point, origin_point, point)
                angle_list.append(angle)

            abs_angle_list = [abs(ele) for ele in angle_list]
            forward_idx = abs_angle_list.index(min(abs_angle_list))

            if self.map_handler.is_segment_assigned(intersection_segment_list[forward_idx]["guid"]) is False:
                forward_road = Road(self.map_handler, intersection_segment_list[forward_idx])
                self.map_handler.road_list.append(forward_road)
                self.forwardRoad = forward_road
                angle_list.pop(forward_idx)

            if len(angle_list) == 2:
                if (angle_list[0] > 0 and angle_list[1] > 0) or (angle_list[0] < 0 and angle_list[1] < 0):
                    print(point_list)
                    print(angle_list)

            # for i in range(len(angle_list)):
            #     if i == forward_idx:
            #
            #


    def add_segment_to_road(self, segment, order):
        self.add_points(segment, order)
        self.segment_list.append(segment["guid"])
        self.map_handler.assign_segment_to_road(segment, self.id)

    def add_points(self, segment, order):
        geometry = segment["geometry"]
        px_values = [point["px"] for point in geometry]
        py_values = [point["py"] for point in geometry]
        tx_values = [point["tx"] for point in geometry]
        ty_values = [point["ty"] for point in geometry]
        width = segment["geometry"][0]["width"]

        # Calculate perpendicular vectors to represent the road boundaries
        normal_x = np.array([-ty for ty in ty_values])
        normal_y = np.array([tx for tx in tx_values])

        norm = np.sqrt(normal_x ** 2 + normal_y ** 2)
        normal_x /= norm
        normal_y /= norm

        if order == "next":
            for i in range(len(px_values)):
                self.left_points.append(
                    (px_values[i] - (width / 2) * normal_x[i], py_values[i] - (width / 2) * normal_y[i]))
                self.right_points.append(
                    (px_values[i] + (width / 2) * normal_x[i], py_values[i] + (width / 2) * normal_y[i]))
        elif order == "previous":
            for i in range(len(px_values) - 1, -1, -1):
                self.left_points.insert(0,
                                        (px_values[i] - (width / 2) * normal_x[i],
                                         py_values[i] - (width / 2) * normal_y[i]))
                self.right_points.insert(0,
                                         (px_values[i] + (width / 2) * normal_x[i],
                                          py_values[i] + (width / 2) * normal_y[i]))


class MapHandler(object):
    def __init__(self):
        with open('/home/lucas/Documents/UTBM/VA50/catkin_ws/src/road_network.json', 'r') as f:
            self.road_network = json.load(f)
        self.lane_width = 3.5

        self.segment_dict = {}
        self.road_list = []

        self.left_boundaries_x = []
        self.left_boundaries_y = []
        self.right_boundaries_x = []
        self.right_boundaries_y = []

        self.process_road_network()

        j = 0
        for i in range(len(self.road_network)):
            if self.is_segment_assigned(self.road_network[i]["guid"]) is False:
                self.road_list.append(Road(self, self.road_network[i]))
                j += 1
        # self.road_list.append(Road(self, self.road_network[3]))
        print(len(self.road_list))
        print(j)
        self.prev_x = 0
        self.prev_y = 0
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.plot_road()

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

    def merge_roads(self, old_road, new_road_id, position_to_merge):
        new_road = None
        for road in self.road_list:
            if road.id == new_road_id:
                new_road = road

        if position_to_merge == "previous":
            print("new road")
            # print(old_road.segment_list)
            # print(new_road.segment_list)

            new_road.segment_list[:0] = old_road.segment_list
            new_road.left_points[:0] = old_road.left_points
            new_road.right_points[:0] = old_road.right_points
            # print(new_road.segment_list)
            # print("--------------")
            #
            # del old_road

    def get_lane_side(self, object):
        if np.abs(object.x) <= self.lane_width / 2 and object.z >= 0:
            return lane_side.FRONT
        elif object.x < -self.lane_width / 2 and object.x >= -self.lane_width * 3 / 2:
            return lane_side.LEFT
        elif object.x > self.lane_width / 2 and object.x <= self.lane_width * 3 / 2:
            return lane_side.RIGHT
        else:
            return lane_side.AWAY

    def is_on_road(self, target_position):
        '''Return true if target_position is on the road
            - target_position: 2d-tuple (x,y)
        '''
        for left_boundary_x, left_boundary_y, right_boundary_x, right_boundary_y in zip(self.left_boundaries_x,
                                                                                        self.left_boundaries_y,
                                                                                        self.right_boundaries_x,
                                                                                        self.right_boundaries_y):
            # Check if the point is within the bounding box of the road segment
            for i in range(len(left_boundary_x) - 1):
                if min(left_boundary_x[i], right_boundary_x[i], left_boundary_x[i + 1], right_boundary_x[i + 1]) <= \
                        target_position[0] <= max(left_boundary_x[i], right_boundary_x[i], left_boundary_x[i + 1],
                                                  right_boundary_x[i + 1]) and \
                        min(left_boundary_y[i], right_boundary_y[i], left_boundary_y[i + 1], right_boundary_y[i + 1]) <= \
                        target_position[1] <= max(left_boundary_y[i], right_boundary_y[i], left_boundary_y[i + 1],
                                                  right_boundary_y[i + 1]):
                    return True
        return False

    def process_road_network(self):
        '''Return true if target_position is on the road
            - target_position: 2d-tuple (x,y)
        '''
        for segment in self.road_network[0:2]:
            geometry = segment["geometry"]
            px_values = [point["px"] for point in geometry]
            py_values = [point["py"] for point in geometry]
            tx_values = [point["tx"] for point in geometry]
            ty_values = [point["ty"] for point in geometry]
            width = segment["geometry"][0]["width"]

            # Calculate perpendicular vectors to represent the road boundaries
            normal_x = np.array([-ty for ty in ty_values])
            normal_y = np.array([tx for tx in tx_values])

            norm = np.sqrt(normal_x ** 2 + normal_y ** 2)
            normal_x /= norm
            normal_y /= norm

            # Calculate road boundaries
            self.left_boundaries_x.append(px_values - (width / 2) * normal_x)
            self.left_boundaries_y.append(py_values - (width / 2) * normal_y)
            self.right_boundaries_x.append(px_values + (width / 2) * normal_x)
            self.right_boundaries_y.append(py_values + (width / 2) * normal_y)

    def plot_road(self):
        for i, road in enumerate(self.road_list):
            if road.display:
                if len(road.segment_list) == 1:
                    color = "b"
                    x_left, y_left = zip(*road.left_points)
                    x_right, y_right = zip(*road.right_points)

                    # Tracer le graphique
                    self.ax.plot(x_left, y_left, linestyle='-', color=color)
                    self.ax.plot(x_right, y_right, linestyle='-', color=color)
                else:
                    color = colors[i % len(colors)]
                    x_left, y_left = zip(*road.left_points)
                    x_right, y_right = zip(*road.right_points)

                    # Tracer le graphique
                    self.ax.plot(x_left, y_left, linestyle='-', color=color)
                    self.ax.plot(x_right, y_right, linestyle='-', color=color)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Road Visualization')
        plt.axis('equal')
        plt.show()

    # DO NOT USE : debug function
    def on_click(self, event):
        pass

    # # Extract the clicked coordinates
    # x, y = event.xdata, event.ydata
    # print(f"prev_x={self.prev_x}, prev_y={self.prev_y}")
    # print(f"x={x}, y={y}")

    # res = self.are_points_on_same_segment(self.prev_x, self.prev_y, x, y, self.road_network)
    # print(f"res={res}")
    # self.prev_x = x
    # self.prev_y = y

    # DO NOT USE : debug function
    def are_points_on_same_segment(self, x1, y1, x2, y2, road_data):
        for segment in road_data:
            geometry = segment["geometry"]
            px_values = [point["px"] for point in geometry]
            py_values = [point["py"] for point in geometry]
            tx_values = [point["tx"] for point in geometry]
            ty_values = [point["ty"] for point in geometry]
            width = segment["geometry"][0]["width"]

            # Calculate perpendicular vectors to represent the road boundaries
            dx = np.array([-ty for ty in ty_values])
            dy = np.array([tx for tx in tx_values])

            # Normalize the vectors
            length = np.sqrt(dx ** 2 + dy ** 2)
            dx /= length
            dy /= length

            # Calculate road boundaries
            left_boundary_x = px_values - (width / 2) * dx
            left_boundary_y = py_values - (width / 2) * dy
            right_boundary_x = px_values + (width / 2) * dx
            right_boundary_y = py_values + (width / 2) * dy

            # Check if both points are within the bounding box of the road segment
            if (min(left_boundary_x) <= x1 <= max(right_boundary_x) and
                    min(left_boundary_y) <= y1 <= max(right_boundary_y) and
                    min(left_boundary_x) <= x2 <= max(right_boundary_x) and
                    min(left_boundary_y) <= y2 <= max(right_boundary_y)):
                return True

        return False


def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    angle = np.arctan2(np.cross(ba, bc), np.dot(ba, bc))

    return angle

if __name__ == "__main__":
    node = MapHandler()