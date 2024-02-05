# ros-autopilot
This project provides a set of ROS packages for perception and decision-making for autonomous driving in a dynamic environment. This project assumes that a structural map describing the static environment such as roads is available. In our test case, this map is described in the file `road_network.json`.
## Packages usage
### perception
The perception package goal is to detect and extract the position of the surrounding objects on the road. The perception is based on 2 types of sensors: RGB Fisheye Camera and LiDAR.
```sh
Usage: perception_core.py <config-file> [options]
    -h, --help      print this message
    --rviz          publish perception visualisationZ
    --verbose       verbose mode, print detected objects
    --yolov8l       use yolov8l model for better accuracy
    --use-map       use structural map to filter out object not on the road
```
### decision
The decision package makes decisions based on the environment perceived by the vehicle. It can adapt the vehicle's speed to the vehicle in front, stop if necessary and overtake if the situation allows.
```sh
Usage: decision_core.py <config-file> [<refresh-rate>]
```
### minimap
The minimap package provides a simplified visualization of perception and decision-making using a minimap.
```sh
Usage: map_plotter.py <config-file>
```
## Build
### Requirements
Install dependencies
```
pip install -r requirements.txt
```
### With catkin
```sh
mkdir -p catkin_ws/src
git clone https://github.com/gabswb/ros-autopilot.git catkin_ws/src
cd catkin_ws/
catkin_make
```
### Run the project
Start roscore
```sh
roscore
```
Launch perception package
```
rosrun perception perception_core.py src/config/config-file.yml [options]
```
Launch decision package
```
rosrun decision decision_core.py src/config/config-file.yml [<input-frequency>]
```
Launch minimap package
```
rosrun minimap map_plotter.py src/config/config-file.yml
```
