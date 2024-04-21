#!/usr/bin/env python2.7
import carla
import threading
import time
import random
import rospy

import numpy as np
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
import tf2_ros

from sensor import Sensor
from camera import Camera
from lidar import Lidar

import yaml

import math

import sys

_DATATYPES = {PointField.INT8: ('b', 1), PointField.UINT8: ('B', 1), PointField.INT16: ('h', 2),
              PointField.UINT16: ('H', 2), PointField.INT32: ('i', 4), PointField.UINT32: ('I', 4),
              PointField.FLOAT32: ('f', 4), PointField.FLOAT64: ('d', 8)}


class CarlaScenario(object):
    def __init__(self, config, decision=False, car_forward=False, car_backward=False):
        self.config = config
        self.decision = decision
        self.car_forward = car_forward
        self.car_backward = car_backward

        # Get CARLA's world
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.reset_world()

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Spawn actors on the world
        spawn_point = carla.Transform(carla.Location(x=45.018200, y=133.947205, z=0.6),
                                      carla.Rotation(pitch=0.000000, yaw=-179.679535, roll=0.000000))
        self.my_vehicle, self.camera, self.lidar = self.instantiate_vehicle(spawn_point, color='150,0,0',
                                                                            with_sensors=True)

        self.tf_broadcaster.sendTransform([self.lidar.tf, self.camera.tf])

        spawn_point_vehicle_warning = carla.Transform(spawn_point.transform(carla.Location(x=25)), spawn_point.rotation)
        self.vehicle_warning = self.instantiate_vehicle(spawn_point_vehicle_warning, color='0,0,150')

        if self.car_forward:
            face_vehicle_spawn_point = carla.Transform(carla.Location(x=-80.018200, y=137.6, z=0.6),
                                                       carla.Rotation(pitch=0.000000, yaw=-0, roll=0.000000))
            self.face_vehicle = self.instantiate_vehicle(face_vehicle_spawn_point, color='0,150,0')

        if self.car_backward:
            backward_vehicle_spawn_point = carla.Transform(carla.Location(x=50, y=137.6, z=0.6),
                                                           carla.Rotation(pitch=0.000000, yaw=-180, roll=0.000000))
            self.backward_vehicle = self.instantiate_vehicle(backward_vehicle_spawn_point, color='150,150,0')

        # Wait everything is well instantiate then move spectator
        time.sleep(1)

        self.move_spectator(spawn_point)

        # Start scenario thread
        scenario = threading.Thread(target=self.thread_scenario)
        scenario.start()

        rospy.loginfo(f"Carla scenario started with parameters:")
        rospy.loginfo(f"\tCar taking decisions: {self.decision}")
        rospy.loginfo(f"\tOther car coming in front of us: {self.car_forward}")
        rospy.loginfo(f"\tOther car overtaking us: {self.car_backward}")

    def reset_world(self):
        self.world = self.client.reload_world()

    def instantiate_vehicle(self, spawn, color='0,0,0', with_sensors=False):
        vehicle_bp = self.bp_lib.find('vehicle.lincoln.mkz_2020')
        vehicle_bp.set_attribute('color', color)
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn)
        if with_sensors:
            camera_bp = self.bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '1028')
            camera_bp.set_attribute('image_size_y', '771')
            camera_bp.set_attribute('sensor_tick', '0.03')
            # camera_bp.set_attribute('lens_k', '1')
            # camera_bp.set_attribute('lens_circle_falloff', '5')
            # camera_bp.set_attribute('lens_circle_multiplier', '1')
            # camera_bp.set_attribute('lens_y_size', '0.35')
            # camera_bp.set_attribute('lens_x_size', '0.35')
            camera_init_trans = carla.Transform(carla.Location(x=0.4, z=1.6))
            v_camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

            camera_class = Camera(v_camera, "middle-lidar", self.config["map"]["reference-frame"],
                                  self.config["topic"]["forward-camera"], self.config["topic"]["forward-camera-info"],
                                  0)

            lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '50.0')
            lidar_bp.set_attribute('noise_stddev', '0.1')
            lidar_bp.set_attribute('upper_fov', '15.0')
            lidar_bp.set_attribute('lower_fov', '-20.0')
            lidar_bp.set_attribute('channels', '256.0')
            lidar_bp.set_attribute('rotation_frequency', '360.0')
            lidar_bp.set_attribute('points_per_second', '1500000')

            lidar_init_trans = carla.Transform(carla.Location(x=0.4, z=1.6))
            v_lidar = self.world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)

            lidar_class = Lidar(v_lidar, "base_link", "middle-lidar",
                                self.config["topic"]["pointcloud"], 1)

            return vehicle, camera_class, lidar_class
        else:
            return vehicle

    def get_random_spawn_point(self):
        spawn_points = self.world.get_map().get_spawn_points()
        point = random.choice(spawn_points)

        return point

    def move_spectator(self, point):
        spectator = self.world.get_spectator()
        point = carla.Transform(carla.Location(x=-30.754190, y=147.055908, z=6.182383),
                                carla.Rotation(pitch=-15.347803, yaw=-35.719940, roll=0.000000))
        spectator.set_transform(point)

    def thread_scenario(self):
        accelerate_vehicle(self.my_vehicle, 0.64)
        accelerate_vehicle(self.vehicle_warning, 1)

        time.sleep(3)

        stop_vehicle(self.vehicle_warning)

        if not self.decision:
            time.sleep(3)
            stop_vehicle(self.my_vehicle)

        if self.car_forward:
            accelerate_vehicle(self.face_vehicle, 1.0)

        if self.car_backward:
            accelerate_vehicle(self.backward_vehicle, 1.0)

        time.sleep(3)
        turn_warning_on(self.vehicle_warning)

        if not self.decision:
            time.sleep(1.5)

            change_way = threading.Thread(target=thread_change_way, args=(self.my_vehicle,))
            change_way.start()

        time.sleep(10)
        turn_lights_off(self.vehicle_warning)


def accelerate_vehicle(vehicle, value):
    vehicle.apply_control(carla.VehicleControl(throttle=value, brake=0.0))


def stop_vehicle(vehicle):
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))


def thread_change_way(vehicle):
    time.sleep(1.5)
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.7))
    time.sleep(2)
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
    time.sleep(1.65)
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.5))
    time.sleep(0.74)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))


def turn_warning_on(vehicle):
    vehicle.set_light_state(
        carla.VehicleLightState(carla.VehicleLightState.RightBlinker | carla.VehicleLightState.LeftBlinker))


def turn_lights_off(vehicle):
    vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            f"Usage : {sys.argv[0]} <config-file> [--decision]  [--car-forward] [--car-backward]")
    else:
        with open(sys.argv[1], "r") as config_file:
            config = yaml.load(config_file, yaml.Loader)

        carla_node = rospy.init_node("carla")
        p = CarlaScenario(config, '--decision' in sys.argv,
                          '--car-forward' in sys.argv,
                          '--car-backward' in sys.argv)
        rospy.spin()
