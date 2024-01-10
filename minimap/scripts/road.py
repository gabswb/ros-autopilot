import numpy as np
import itertools


class Road():
    id_obj = itertools.count()

    def __init__(self, map_handler, first_segment):
        self.id = next(Road.id_obj)
        self.segment_list = []

        self.left_points = []
        self.path_to_follow = []
        self.right_points = []

        self.oppositeRoad = None
        self.leftRoad = None
        self.forwardRoad = None
        self.rightRoad = None
        self.previous_road_list = []

        self.display = True

        self.road_network = map_handler.road_network
        self.map_handler = map_handler

        self.map_handler.road_list.append(self)
        self.find_other_segments(first_segment)

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
                if len(previous_segment["next"]) == 1:
                    self.add_segment_to_road(previous_segment, "previous")
                else:
                    break
            else:
                break

        while len(next_segment["next"]) == 1:
            segment_road = self.map_handler.is_segment_assigned(next_segment["next"][0])
            if segment_road is False:
                next_segment_guid = next_segment["next"][0]
                next_segment = self.map_handler.get_segment(next_segment_guid)
                if len(next_segment["previous"]) > 1:
                    new_road = Road(self.map_handler, next_segment)
                    self.forwardRoad = new_road
                    self.forwardRoad.previous_road_list.append(self)
                    return
                else:
                    self.add_segment_to_road(next_segment, "next")
            else:
                forward_road = self.map_handler.get_road(next_segment["next"][0])
                self.forwardRoad = forward_road
                self.forwardRoad.previous_road_list.append(self)
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

            # print(self.forwardRoad)
            if self.map_handler.is_segment_assigned(intersection_segment_list[forward_idx]["guid"]) is False:
                forward_road = Road(self.map_handler, intersection_segment_list[forward_idx])
                # self.map_handler.road_list.append(forward_road)
                self.forwardRoad = forward_road
                self.forwardRoad.previous_road_list.append(self)
            else:
                forward_road = self.map_handler.get_road(intersection_segment_list[forward_idx]["guid"])
                self.forwardRoad = forward_road
                if self not in self.forwardRoad.previous_road_list:
                    self.forwardRoad.previous_road_list.append(self)
            # print("---")

            for i in range(len(angle_list)):
                if i == forward_idx:
                    pass
                elif angle_list[i] > 0:
                    if self.map_handler.is_segment_assigned(intersection_segment_list[i]["guid"]) is False:
                        left_road = Road(self.map_handler, intersection_segment_list[i])
                        if self.leftRoad is not None:
                            self.leftRoad = left_road
                            self.leftRoad.previous_road_list.append(self)
                        else:
                            self.leftRoad = left_road
                            self.leftRoad.previous_road_list.append(self)
                    else:
                        left_road = self.map_handler.get_road(intersection_segment_list[forward_idx]["guid"])
                        self.leftRoad = left_road
                        if self not in self.leftRoad.previous_road_list:
                            self.leftRoad.previous_road_list.append(self)

                elif angle_list[i] < 0:
                    if self.map_handler.is_segment_assigned(intersection_segment_list[i]["guid"]) is False:
                        right_road = Road(self.map_handler, intersection_segment_list[i])
                        if self.rightRoad is not None:
                            self.rightRoad = right_road
                            self.rightRoad.previous_road_list.append(self)
                        else:
                            self.rightRoad = right_road
                            self.rightRoad.previous_road_list.append(self)
                    else:
                        right_road = self.map_handler.get_road(intersection_segment_list[forward_idx]["guid"])
                        self.rightRoad = right_road
                        if self not in self.rightRoad.previous_road_list:
                            self.rightRoad.previous_road_list.append(self)

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
                self.path_to_follow.append((px_values[i], py_values[i]))
                self.right_points.append(
                    (px_values[i] + (width / 2) * normal_x[i], py_values[i] + (width / 2) * normal_y[i]))
        elif order == "previous":
            for i in range(len(px_values) - 1, -1, -1):
                self.left_points.insert(0,
                                        (px_values[i] - (width / 2) * normal_x[i],
                                         py_values[i] - (width / 2) * normal_y[i]))
                self.path_to_follow.insert(0, (px_values[i], py_values[i]))
                self.right_points.insert(0,
                                         (px_values[i] + (width / 2) * normal_x[i],
                                          py_values[i] + (width / 2) * normal_y[i]))

def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    angle = np.arctan2(np.cross(ba, bc), np.dot(ba, bc))

    return angle