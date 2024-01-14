# Perception of Surrounding Environment for Navigation Decision of Autonomous Vehicles
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
# launch ros
roscore
# launch utac simulator
simulator 
# launch percepetion node
rosrun perception perception_core.py src/config-utac.yml 
    --rviz # publish on visuzalition topic 
    --lidar-projection # !! only for visualization purpuse 
    --log-objects # log published objects (yolo detection + its distance + its instance ID)
    --time-statistics # time needed for the main operation (yolo detection time, distance extraction time, ...)
    --yolov8l # use yolov8l model for better accuracy
    --use-map # use structural map to filer out object not on the road
    --no-lights # disable lights detection (for better performance)
    --only-front-camera # use only camera in perception pipeline
# launch decision node
rosrun decision decision_core.py src/config-utac.yml 100 # 100Hz = publishing control input frequency (need adjustments depending on the computer computing capacity)
# launch visiualization node
rosrun minimap map_plotter.py src/config-utac.yml                                         
```
