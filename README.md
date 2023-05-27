# Persian Plate Recognition
Recognize Persian plate with YOLOv8

This Repo created for detect persian cars and plates and then recognize every persian characters on the plate.

## Prerequisite
yolov8 Ultralytics and all of Requirements for yolo8


## Demo

https://github.com/shahabbai/PersianPlateRecog/assets/133869713/3e4b013d-1eba-4896-8a2f-0e74a99535e1


## Installation
```
pip install ultralytics==8.0.104
```

## Run
3 options for run:

1.use the main python script
```
python main.py
```

2.use my streamlit link on your browser(this option no need any installation):

[Streamlit](https://shahabbai-persianplaterecog-app-84qgn4.streamlit.app/)

3.use streamlit app.py and run it locally on your pc:

```
pip install streamlit
streamlit run app.py
```
## Datasets
Created two datasets :

1. Dataset for detection cars and plates [Link](https://universe.roboflow.com/shahab-jafari-1vorv/persian-car)

2. Dataset for detection chars of the plates [Link](https://universe.roboflow.com/shahab-jafari-1vorv/persian-plate-characters-mvinj)
## Models
for simplicity of computational using yolov8s for cars and plates detection and using yolov8n for character detection
## Training Results
1. yolov8s model for cars and plates detection

![results (1)](https://github.com/shahabbai/PersianPlateRecog/assets/133869713/8cb0e04b-edc9-4f2a-b560-3daec538af6c)

2. yolov8n model for characters detection



![yolov8n_char_new_small](https://github.com/shahabbai/PersianPlateRecog/assets/133869713/59db56cf-94a4-4289-ad60-b8f58225b7c2)

