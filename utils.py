#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import config


charclassnames = ['0','9','b','d','ein','ein','g','gh','h','n','s','1','malul','n','s','sad','t','ta','v','y','2'
                  ,'3','4','5','6','7','8']



def _display_detected_frames(conf, model_object, model_char, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    #image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res_object = model_object.predict(image, conf=conf)
    for i in res_object:
        bbox = i.boxes
        for box in bbox:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            #confs = math.ceil((box.conf[0]*100))/100
            cls_names = int(box.cls[0])
            

            #check plate to recognize characters with yolov8n model
            if cls_names == 1:
                char_display = []
                #crop plate from frame
                plate_img = image[y1:y2, x1:x2]
                #plate_img = uploaded_image[y1:y2, x1:x2]
                #detect characters of plate with yolov8n model
                plate_output = model_char(plate_img, conf=0.4)
                
                #extract bounding box and class names
                bbox = plate_output[0].boxes.xyxy
                cls = plate_output[0].boxes.cls
                #make a dict and sort it from left to right to show the correct characters of plate
                keys = cls.numpy().astype(int)
                values =bbox[:, 0].numpy().astype(int)
                dictionary = list(zip(keys, values))
                sorted_list = sorted(dictionary, key=lambda x: x[1])
                #convert all characters to a string
                for i in sorted_list:
                    char_class = i[0]
                    #char_display.append(plate_output[0].names[char_class])
                    char_display.append(charclassnames[char_class])
                char_result ='Plate: ' + (''.join(char_display))
    
                #just show the correct characters in output
    
    inText = 'Vehicle In'
    outText = 'Vehicle Out'
    if config.OBJECT_COUNTER1 != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER1.items()):
            inText += ' - ' + str(key) + ": " +str(value)
    if config.OBJECT_COUNTER != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER.items()):
            outText += ' - ' + str(key) + ": " +str(value)
    
    st.markdown(
    f'<style>img {{ max-width: {640}px; height: auto; }}</style>',
    unsafe_allow_html=True
)
    # Plot the detected objects on the video frame
    st_count.write(inText + '\n\n' + outText)
    res_plotted = res_object[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    text_placeholder = st.empty()
    text_placeholder.text(char_result)
    #st.write(char_result)


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model_object, model_char):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res_object = model_object.predict(uploaded_image,
                                    conf=conf)
                boxes = res_object[0].boxes
                    #extract bounding box and class names
                res_plotted = res_object[0].plot()[:, :, ::-1]    
                for i in res_object:
                    bbox = i.boxes
                    for box in bbox:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        #confs = math.ceil((box.conf[0]*100))/100
                        cls_names = int(box.cls[0])
                        

                        #check plate to recognize characters with yolov8n model
                        if cls_names == 1:
                            char_display = []
                            #crop plate from frame
                            plate_img = uploaded_image.crop((x1, y1, x2, y2))
                            #plate_img = uploaded_image[y1:y2, x1:x2]
                            #detect characters of plate with yolov8n model
                            plate_output = model_char(plate_img, conf=0.4)
                            
                            #extract bounding box and class names
                            bbox = plate_output[0].boxes.xyxy
                            cls = plate_output[0].boxes.cls
                            #make a dict and sort it from left to right to show the correct characters of plate
                            keys = cls.numpy().astype(int)
                            values =bbox[:, 0].numpy().astype(int)
                            dictionary = list(zip(keys, values))
                            sorted_list = sorted(dictionary, key=lambda x: x[1])
                            #convert all characters to a string
                            for i in sorted_list:
                                char_class = i[0]
                                #char_display.append(plate_output[0].names[char_class])
                                char_display.append(charclassnames[char_class])
                            char_result ='Plate: ' + (''.join(char_display))
                
                            #just show the correct characters in output
                

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    if len(char_display) == 8:
                        st.write(char_result)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model_object, model_char):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.markdown(
    f'<style>video {{ width: {640}px !important; height: auto !important; }}</style>',
    unsafe_allow_html=True
)
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_count = st.empty()
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model_object,model_char,
                                                     st_count,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model_object, model_char):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_count = st.empty()
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model_object,
                    st_count,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
