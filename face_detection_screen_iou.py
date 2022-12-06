from __future__ import division
import cv2
import time
import sys
#from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera


conf_threshold = 0.7

source = 0
line_length = 20
screen_box_dist_x = 130
screen_box_dist_y = 120
iou_threshold = 0.5

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bbox = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bbox.append([x1, y1, x2, y2])
            
    return frameOpencvDnn, bbox


def mask_prediction(frameOpencvDnn, bbox):
    x1, y1, x2, y2 = bbox
    img = frameOpencvDnn[y1+20:y2+50, x1+10:x2+10]
    image = cv2.resize(img, (64, 64))            
    image = image.astype("float") / 255.0
    image = np.asarray(image)
    #image = img_to_array(image)
    image = image.transpose(2,0,1)
    image = np.expand_dims(image, axis=0)

    classifier.setInput(image)
    [(nomask, mask)] = classifier.forward()
    
    label = "Mask" if mask > 0.75 else "No Mask"
    return label


def compute_screen_box(frame):
    #Screen Box Coordinates
    center_x = int(frame.shape[1]/2)
    center_y = int(frame.shape[0]/2)

    screen_x1 = center_x - screen_box_dist_x
    screen_x2 = center_x - screen_box_dist_x
    screen_x3 = center_x + screen_box_dist_x
    screen_x4 = center_x + screen_box_dist_x
    screen_y1 = center_y - screen_box_dist_y
    screen_y2 = center_y + screen_box_dist_y
    screen_y3 = center_y - screen_box_dist_y
    screen_y4 = center_y + screen_box_dist_y
    screen_coords = [screen_x1, screen_x2, screen_x3, screen_x4, screen_y1, screen_y2, screen_y3, screen_y4]
    return screen_coords

def check_iou(img, face, screen_coords, temp):
    screen_x1, screen_x2, screen_x3, screen_x4, screen_y1, screen_y2, screen_y3, screen_y4 = screen_coords
    screen_box_area = (screen_x4 - screen_x2) * (screen_y2 - screen_y1)
    cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (0, 255, 255), 1)
    if face[0] >= screen_x1 and face[1] >= screen_y1 and face[2] <= screen_x3 and face[3] <= screen_y2:
        face_width = face[2] - face[0]
        face_height = face[3] - face[1]
        face_box_area = face_width * face_height 

        if face_box_area/screen_box_area >= iou_threshold:
            color = (0, 255, 0)
            instruction = 'Face is Aligned!'
            
    
            out_label = mask_prediction(outOpencvDnn, face)
            if out_label == "Mask":
                label_color = (0, 255, 0) 
            else:
                label_color = (0, 0, 255) 
            cv2.putText(img, out_label, (screen_x2,screen_y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2, cv2.LINE_AA)
            if temp <= 99.0:
                temp_color = (0, 255, 0) 
            else:
                temp_color = (0, 0, 255) 
            cv2.putText(img, "Temperature: %.2f"%temp+" F", (screen_x2,screen_y2+60), cv2.FONT_HERSHEY_SIMPLEX, 1, temp_color, 2, cv2.LINE_AA)
        else:
            color = (0, 0, 255)
            instruction = 'Please Bring Your Face Nearer to the Camera!'
    else:
        color = (0, 0, 255) 
        instruction = 'Please Align Your Face in Given Region...'

    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.5
    thickness = 4
    font_thickness = 2
    img = cv2.putText(img, instruction, (screen_x1,screen_y1-5), font,  
                       fontScale, color, font_thickness, cv2.LINE_AA) 

    cv2.line(img, (screen_x1, screen_y1), (screen_x1 , screen_y1 + line_length), color, thickness)  #-- top-left
    cv2.line(img, (screen_x1, screen_y1), (screen_x1 + line_length , screen_y1), color, thickness)

    cv2.line(img, (screen_x2, screen_y2), (screen_x2 , screen_y2 - line_length), color, thickness)  #-- bottom-left
    cv2.line(img, (screen_x2, screen_y2), (screen_x2 + line_length , screen_y2), color, thickness)

    cv2.line(img, (screen_x3, screen_y3), (screen_x3 - line_length, screen_y3), color, thickness)  #-- top-right
    cv2.line(img, (screen_x3, screen_y3), (screen_x3, screen_y3 + line_length), color, thickness)

    cv2.line(img, (screen_x4, screen_y4), (screen_x4 , screen_y4 - line_length), color, thickness)  #-- bottom-right
    cv2.line(img, (screen_x4, screen_y4), (screen_x4 - line_length , screen_y4), color, thickness)

    return img
    

if __name__ == "__main__" :

    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    DNN = "TF"
    if DNN == "CAFFE":
        modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    
    class_model = "weights/tf_model.pb"
    #class_conf = "/home/rupali/rr/mask_or_not_corona_keras_dnn/image-classification-keras/model/tf_model.pbtxt"
    classifier = cv2.dnn.readNetFromTensorflow(class_model) #,class_conf)    

    """
    # raspberry pi 3
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 8
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.1)
    """
    
    #frame = cv2.imread('/home/rupali/rr/cctv_facemask/images/test/08hongkong-briefing02-videoSixteenByNine3000-v5.jpg')
    #vid_writer = cv2.VideoWriter('output-dnn-2.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))


    frame_count = 0
    tt_opencvDnn = 0

    """
    # -- usb 3.0 camera
    source = 1
    cap = cv2.VideoCapture('/dev/video0')
    hasFrame, frame = cap.read()
    """

    # -- raspberry pi 3 camera module
    live_frame = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
    
    
    while(True):
        # -- raspberry pi 3 camera module
        frame = next(live_frame).array 
        
        """ -- usb 3.0 camera module
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        
        """
    
        print(frame.shape)
        
        #time.sleep(0.1)
        frame_count += 1

        if frame_count == 1:
            screen_coords = compute_screen_box(frame)

        t = time.time()
        outOpencvDnn, bbox = detectFaceOpenCVDnn(net,frame)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn

        temp = np.random.uniform(96,104)

        if len(bbox) > 0:
            img = check_iou(frame, bbox[0], screen_coords, temp)
            
            #vid_writer.write(img)
            if frame_count == 1:
                tt_opencvDnn = 0

            cv2.imshow("screen", img)
            k = cv2.waitKey(10)
            if k == 27:
                break
        #rawCapture.truncate(0)
             
    cv2.destroyAllWindows()
    #vid_writer.release()
    
    
