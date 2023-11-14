# ros-autopilot
## Build
### Requirements
- install packages listed in `requirements.txt`
### With catkin
```sh
mkdir -p catkin_ws/src
git clone https://github.com/gabswb/ros-autopilot.git catkin_ws/src
cd catkin_ws/
catkin_make
```

### Usage
Run at the root dir (the one with src/ build/ etc ...)  :
```sh
roscore
simulator # launch utac simulator
rosrun perception perception_core.py src/config-utac.yml -v --no-publish --lidar-projection
```