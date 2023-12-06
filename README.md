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
rosrun perception perception_core.py src/config-utac.yml 
    -v # open a opencv window with the forward camera perspective
    --rviz # publish on visuzalition topic 
    --lidar-projection # !! only for visualization purpuse 
    --log-objects # log published objects (yolo detection + its distance + its instance ID)
    --time-statistics # time needed for the main operation (yolo detection time, distance extraction time, ...)
    --yolov8l # use yolov8l model for better accuracy
    --use-map # use structural map to filer out object not on the road
```

### ONNX model generation
> ONNX is an open standard format for representing machine learning models. Needed by opencv (spare torch usage)
```bash
# Clone the repository. 
git clone https://github.com/ultralytics/YOLOv5
 
cd YOLOv5 # Install dependencies.
pip install -r requirements.txt
pip install onnx
 
# Download .pt model.
wget https://github.com/ultralytics/YOLOv5/releases/download/v6.1/YOLOv5n.pt
python export.py --weights models/YOLOv5n.pt --include onn --imgsz 416 416 --simplify --opset 11
```