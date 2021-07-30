from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
#
import threading
import ini
import datetime
import json
from Connserver import Connserver
####
from time import sleep 
from threading import Thread 
##from pynput import keyboard
#wa
"""
def convertBack2(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes2(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack2(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
	cv2.circle(img, (int(x),int(y)), 1, (255,0,0), 2)#
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img
"""
def init():
   global preLeft,preRight,LCheck,RCheck
   global LeftNow,RightNow
   global LeftCount,RightCount 
   global forward_temp, forward_appear  
   global forward_sum_threshold, forward_pixel, forward_road_coord, forward_gaussian_range, Fgaussian_range, darknet_road_line, count
   global forward_pixel_threshold, two_way_frist, forward_road_number, count_clear
   global forward_pMOG2
   global forward_foreground
   global Croad, Cline, ExistM, Sspeed, Bspeed, Speed, l_Maxsp, l_Minsp, l_Ct, kname, Csum, count
   global Sp_time1, Sfp_time1, Sp_time2, Sfp_time2, Sp_time3, Sfp_time3, Vsc, Vbc, Svs, Svb, Rhold
   Sspeed = [0, 0, 0]
   Bspeed = [0, 0, 0]
   Vsc = [0, 0, 0]
   Vbc = [0, 0, 0]
   Svs = [0, 0, 0]
   Svb = [0, 0, 0]
   Rhold = [0.0, 0.0, 0.0]
   Croad,Cline = 0, 0
   l_Maxsp = 200
   l_Minsp = 0
   l_Ct = 200
   Speed = 0
   count_clear = 0
   preLeft,preRight,LCheck,RCheck=0,0,0,0
   LeftNow,RightNow,LeftCount,RightCount=0,0,0,0
   global lock
   lock=threading.Lock()

def set_area(x,y,w,h):
    global x_sc,y_sc,w_sc,h_sc
    x_sc,y_sc,w_sc,h_sc=x,y,w,h
def set_count_area(x,y,w,h):
    global c_x,c_y,c_w,c_h
    c_x,c_y,c_w,c_h=x,y,w,h
def set_road_line(x1,y1,x2,y2):
    global road_point1,road_point2
    global _d_x,_d_y,_d_c
    road_point1,road_point2=(x1*416,y1*416),(x2*416,y2*416)
    _d_x,_d_y=(y2-y1),(x1-x2)
    _d_c=-(x1*_d_x+y1*_d_y)
def set_yolo_range(msg):
    global Fgaussian_range, darknet_road_line, count_clear
    global ExistM, forward_road_number, l_Maxsp, l_Minsp, l_Ct
    if(msg=='Error'): return
    msg=msg[:-5]
    #print("debug:",msg)
    #print("_msg=",msg)
    _json=json.loads(msg)
    if(_json['requset'] == 1):
	forward_road_number = _json['road'] + 1
	l_Maxsp = _json['lim_Maxsp']
	l_Minsp = _json['lim_Minsp']
	l_Ct = _json['lim_Ct']
	if(_json['cl'] == 1):
	    count_clear = 1;
	if(_json['road'] == 0):
	    Fgaussian_range[0, 0] = _json['yolo1'][0]
	    Fgaussian_range[0, 1] = _json['yolo1'][1]
	    Fgaussian_range[0, 2] = _json['yolo1'][2]
	    Fgaussian_range[0, 3] = _json['yolo1'][3]
	    darknet_road_line[0, 0] = _json['road1'][0]
	    darknet_road_line[0, 1] = _json['road1'][1]
	    darknet_road_line[0, 2] = _json['road1'][2]
	    darknet_road_line[0, 3] = _json['road1'][3]  #DOWM TO UP
	    darknet_road_line[1, 0] = _json['road2'][0]
	    darknet_road_line[1, 1] = _json['road2'][1]
	    darknet_road_line[1, 2] = _json['road2'][2]
	    darknet_road_line[1, 3] = _json['road2'][3]        
	elif(_json['road'] == 1):
	    Fgaussian_range[0, 0] = _json['yolo1'][0]
	    Fgaussian_range[0, 1] = _json['yolo1'][1]
	    Fgaussian_range[0, 2] = _json['yolo1'][2]
	    Fgaussian_range[0, 3] = _json['yolo1'][3]
	    Fgaussian_range[1, 0] = _json['yolo2'][0]
	    Fgaussian_range[1, 1] = _json['yolo2'][1]
	    Fgaussian_range[1, 2] = _json['yolo2'][2]
	    Fgaussian_range[1, 3] = _json['yolo2'][3]
	    darknet_road_line[0, 0] = _json['road1'][0]
	    darknet_road_line[0, 1] = _json['road1'][1]
	    darknet_road_line[0, 2] = _json['road1'][2]
	    darknet_road_line[0, 3] = _json['road1'][3]
	    darknet_road_line[1, 0] = _json['road2'][0]
	    darknet_road_line[1, 1] = _json['road2'][1]
	    darknet_road_line[1, 2] = _json['road2'][2]
	    darknet_road_line[1, 3] = _json['road2'][3]
	    darknet_road_line[2, 0] = _json['road3'][0]
	    darknet_road_line[2, 1] = _json['road3'][1]
	    darknet_road_line[2, 2] = _json['road3'][2]
	    darknet_road_line[2, 3] = _json['road3'][3]
	elif(_json['road'] == 2):
	    Fgaussian_range[0, 0] = _json['yolo1'][0]
	    Fgaussian_range[0, 1] = _json['yolo1'][1]
	    Fgaussian_range[0, 2] = _json['yolo1'][2]
	    Fgaussian_range[0, 3] = _json['yolo1'][3]
	    Fgaussian_range[1, 0] = _json['yolo2'][0]
	    Fgaussian_range[1, 1] = _json['yolo2'][1]
	    Fgaussian_range[1, 2] = _json['yolo2'][2]
	    Fgaussian_range[1, 3] = _json['yolo2'][3]
	    Fgaussian_range[2, 0] = _json['yolo3'][0]
	    Fgaussian_range[2, 1] = _json['yolo3'][1]
	    Fgaussian_range[2, 2] = _json['yolo3'][2]
	    Fgaussian_range[2, 3] = _json['yolo3'][3]
	    darknet_road_line[0, 0] = _json['road1'][0]
	    darknet_road_line[0, 1] = _json['road1'][1]
	    darknet_road_line[0, 2] = _json['road1'][2]
	    darknet_road_line[0, 3] = _json['road1'][3]
	    darknet_road_line[1, 0] = _json['road2'][0]
	    darknet_road_line[1, 1] = _json['road2'][1]
	    darknet_road_line[1, 2] = _json['road2'][2]
	    darknet_road_line[1, 3] = _json['road2'][3]
	    darknet_road_line[2, 0] = _json['road3'][0]
	    darknet_road_line[2, 1] = _json['road3'][1]
	    darknet_road_line[2, 2] = _json['road3'][2]
	    darknet_road_line[2, 3] = _json['road3'][3]
	    darknet_road_line[3, 0] = _json['road4'][0]
	    darknet_road_line[3, 1] = _json['road4'][1]
	    darknet_road_line[3, 2] = _json['road4'][2]
	    darknet_road_line[3, 3] = _json['road4'][3]	
	if(_json['scooterRoad'] == 1):
	    ExistM = 1
def chk_left(x,y):
    #global _d_x,_d_y,_d_c
    #is_right=false
    #2
    _cc=(-_d_y*(y-road_point1[1]))-(_d_x*(x-road_point1[0]))
    #print("_cc=",_cc)
    if(_cc>0):#/
    	#print("left")
	return True
    else:
	#print("right")
	return False  

#count 
def counting(detections):
    global preLeft,preRight,LCheck,RCheck # L=North R=South 
    global lock
    global LeftNow,RightNow #northnow sourthnow
    global LeftCount,RightCount #northCouter sourthcounter
    #
    Left,Right=0,0
    #
    for detection in detections:
    	x, y, w, h = detection[2][0],detection[2][1],detection[2][2],detection[2][3]
        #
        _is_left=chk_left(x,y) # RIGHT : LEFT
        #    	
	if(_is_left):Left=Left+1
    	else:Right=Right+1
    ## max
    if(Left<preLeft):LCheck=LCheck+1
    else:
	preLeft=Left
	LCheck=0
    if(Right<preRight):RCheck=RCheck+1
    else:
	preRight=Right
	RCheck=0
    #print("count:",Left,":",Right) ##########################
    #	
    lock.acquire()
    LeftNow,RightNow=Left,Right
    if(LCheck>4):
	LeftCount=preLeft-Left
	LCheck=0
	preLeft=Left
    if(RCheck>4):
	RightCount=preRight-Right
	RCheck=0
	preRight=Right
    lock.release()	

############################
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
	
	cv2.circle(img, (int(x),int(y)), 1, (255,0,0), 2)#
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img
#

netMain = None
metaMain = None
altNames = None



def GMM():

    global metaMain, netMain, altNames,frame_resized,darknet_road_line,Fgaussian_range,kname

    configPath = "./cfg/yolov4_t74.cfg"   
    weightPath = "./yolov4-t74_1230_last.weights"    
    metaPath = "./cfg/TESTcoco_old.data"   

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1) 
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #chk cap 
    for kk in range(24):
	    global Csum, count
	    count = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])   
	    kname = str(kk) + ".txt" 
	    cap = cv2.VideoCapture(str(kk) + ".mp4") 
	   
	    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3);
	    cap.set(3, 640)
	    cap.set(4, 480)
	    print("Starting the YOLO loop...")

	    darknet_image = darknet.make_image(darknet.network_width(netMain),
		                            darknet.network_height(netMain),3)
	    while _Keep_Run:

		prev_time = time.time()
		ret, frame_read = cap.read()  

		
		if(not ret):
		    print("Starting the YOLO loop...")
		    print("RTSP error! Please Check Internert")
		    break;
		frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

		yolox=int(frame_rgb.shape[1]*x_sc)
		yoloy=int(frame_rgb.shape[0]*y_sc)
		yolow=int(frame_rgb.shape[1]*w_sc)
		yoloh=int(frame_rgb.shape[0]*h_sc)
		frame_crop=frame_rgb[yoloy:yoloy+yoloh,yolox:yolox+yolow].copy()
		frame_resized0 = cv2.resize(frame_crop, 
		                           (darknet.network_width(netMain),
		                            darknet.network_height(netMain)),
		                           interpolation=cv2.INTER_LINEAR)
	#
		frame_resized=cv2.resize(frame_crop, 
		                           (darknet.network_width(netMain),
		                            darknet.network_height(netMain)),
		                           interpolation=cv2.INTER_LINEAR)
		
		forward_road_number = 3
		
		darknet_road_line = np.array([[0.575,0.448611111111111,0.21796875,0.661111111111111],
			[0.69140625,0.455555555555556,0.3296875,0.780555555555556],
			[0.75,0.45,0.43203125,0.844444444444444],
			[0.8,0.441666666666667,0.621875,0.851388888888889]])
		
		Fgaussian_range = np.array([[0.45,0.543055555555556,0.06484375,0.0513888888888889], 
			[0.578125,0.538888888888889,0.0671875,0.0472222222222222],
			[0.5984375,0.643055555555556,0.08359375,0.0555555555555556]])
		

		count_area = np.array( [ [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0] ] )
		for i in range(forward_road_number):
		    count_area[i, 0]=int(darknet.network_width(netMain) * Fgaussian_range[i, 0])
		    count_area[i, 1]=int(darknet.network_height(netMain) * Fgaussian_range[i, 1])
		    count_area[i, 2]=int(darknet.network_width(netMain) * Fgaussian_range[i, 2])
		    count_area[i, 3]=int(darknet.network_height(netMain) * Fgaussian_range[i, 3])

		
		global count, count_clear, Vsc, Vbc, Bspeed, Sspeed, Rhold  
	    	if (count_clear == 1):    
		    Vsc = [0, 0, 0]
		    Vbc = [0, 0, 0]
		    Bspeed = [0, 0, 0]
		    Sspeed = [0, 0, 0]
		    Rhold = [0, 0, 0]
	      	    for i in range(3):
			for j in range(6):
			    count[i, j] = 0;
	 	    count_clear = 0

		global forward_road_coord
		forward_road_coord = np.array( [ [int(frame_read.shape[1] * Fgaussian_range[0, 0]), int(frame_read.shape[0] * Fgaussian_range[0, 1]), int(frame_read.shape[1] * Fgaussian_range[0, 2]), int(frame_read.shape[0] * Fgaussian_range[0, 3])],[int(frame_read.shape[1] * Fgaussian_range[1, 0]), int(frame_read.shape[0] * Fgaussian_range[1, 1]), int(frame_read.shape[1] * Fgaussian_range[1, 2]), int(frame_read.shape[0] * Fgaussian_range[1, 3])],[int(frame_read.shape[1] * Fgaussian_range[2, 0]), int(frame_read.shape[0] * Fgaussian_range[2, 1]), int(frame_read.shape[1] * Fgaussian_range[2, 2]), int(frame_read.shape[0] * Fgaussian_range[2, 3])] ] )   
		minx, miny, maxw, maxh = forward_road_coord[0, 0], forward_road_coord[0, 1], 0, 0

		for i in range(forward_road_number):
		    if (minx > forward_road_coord[i, 0]):
			minx = forward_road_coord[i, 0]
		    if (miny > forward_road_coord[i, 1]):
			miny = forward_road_coord[i, 1]
		    if (maxh < forward_road_coord[i, 1] + forward_road_coord[i, 3]):
			maxh = forward_road_coord[i, 1] + forward_road_coord[i, 3]
		    if (maxw < forward_road_coord[i, 0] + forward_road_coord[i, 2]):
			maxw = forward_road_coord[i, 0] + forward_road_coord[i, 2]
		forward_gaussian_range = [minx, miny, maxw - minx, maxh - miny]  
	   	global two_way_frist
		global Sp_time1, Sfp_time1, Sp_time2, Sfp_time2, Sp_time3, Sfp_time3
	   	global forward_pMOG2
	   	global forward_foreground
		forward_sum = [ 0, 0, 0, 0 ]
		forward_detect_img = frame_rgb[forward_gaussian_range[1]:forward_gaussian_range[1]+forward_gaussian_range[3] , forward_gaussian_range[0]:forward_gaussian_range[0]+forward_gaussian_range[2]].copy()

		forward_frameGray = cv2.cvtColor(forward_detect_img, cv2.COLOR_BGR2GRAY)
		forward_foreground = forward_frameGray
		forward_pMOG2.apply(forward_frameGray, forward_foreground, 0.009) 

		forward_frameGray2 = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
		forward_foreground2 = forward_frameGray2
		forward_pMOG3.apply(forward_frameGray2, forward_foreground2, 0.009)

		if (two_way_frist < 3):
		    two_way_frist += 1
		if (two_way_frist == 3):
		    if (forward_road_number == 1):           
			global forward_temp, forward_appear 
	   		global forward_sum_threshold, forward_pixel
			forward_road_1 = forward_foreground
	   		forward_road_1 = forward_foreground[forward_road_coord[0, 1] - forward_gaussian_range[1]:forward_road_coord[0, 1] - forward_gaussian_range[1] + forward_road_coord[0, 3],forward_road_coord[0, 0] - forward_gaussian_range[0]:forward_road_coord[0, 0] - forward_gaussian_range[0] + forward_road_coord[0, 2]]
			for i_1 in range(0, forward_road_coord[0, 3]):             
			    for j_1 in range(0, forward_road_coord[0, 2]):		
				forward_pixel[0] = forward_road_1[i_1, j_1]
		                if (forward_pixel[0] > forward_pixel_threshold):
		                    forward_sum[0] += 1                            

		        if (forward_sum[0] > forward_sum_threshold):   
			    Sfp_time1 = datetime.datetime.now().strftime('%f')
			    Sp_time1 = datetime.datetime.now().strftime('%S')
		            forward_appear[0] = 1
		        else:
		            forward_appear[0] = 0
		        if (forward_appear[0] == 0 and forward_temp[0] == 1):  #when car leave gaussain
			    YOLO(frame_resized, darknet_image, 0)
		        if (forward_appear[0] == 1):
		            forward_temp[0] = 1
		        else:
		            forward_temp[0] = 0
		    elif (forward_road_number == 2):      #2 road
		        forward_road_1 = forward_foreground
	   		forward_road_1 = forward_foreground[forward_road_coord[0, 1] - forward_gaussian_range[1]:forward_road_coord[0, 1] - forward_gaussian_range[1] + forward_road_coord[0, 3],forward_road_coord[0, 0] - forward_gaussian_range[0]:forward_road_coord[0, 0] - forward_gaussian_range[0] + forward_road_coord[0, 2]]


			forward_road_2 = forward_foreground
	   		forward_road_2 = forward_foreground[forward_road_coord[1, 1] - forward_gaussian_range[1]:forward_road_coord[1, 1] - forward_gaussian_range[1] + forward_road_coord[1, 3],forward_road_coord[1, 0] - forward_gaussian_range[0]:forward_road_coord[1, 0] - forward_gaussian_range[0] + forward_road_coord[1, 2]]

		        for i_1 in range(forward_road_coord[0, 3]):
		            for j_1 in range(forward_road_coord[0, 2]):
		                forward_pixel[0] = forward_road_1[i_1, j_1]

		                if (forward_pixel[0] > forward_pixel_threshold):
		                    forward_sum[0] += 1


		        for i_2 in range(forward_road_coord[1, 3]):
		            for j_2 in range(forward_road_coord[1, 2]):
		                forward_pixel[1] = forward_road_2[i_2, j_2]

		                if (forward_pixel[1] > forward_pixel_threshold):
		                    forward_sum[1] += 1

		        if (forward_sum[0] > forward_sum_threshold):
		            forward_appear[0] = 1
			    Sfp_time1 = datetime.datetime.now().strftime('%f')
			    Sp_time1 = datetime.datetime.now().strftime('%S')
		        else:
		            forward_appear[0] = 0
		        if (forward_sum[1] > forward_sum_threshold):
		            forward_appear[1] = 1
			    Sfp_time2 = datetime.datetime.now().strftime('%f')
			    Sp_time2 = datetime.datetime.now().strftime('%S')
		        else:
		            forward_appear[1] = 0
		        if (forward_appear[0] == 0 and forward_temp[0] == 1):
		            YOLO(frame_resized, darknet_image, 0)
		        if (forward_appear[1] == 0 and forward_temp[1] == 1):
		            YOLO(frame_resized, darknet_image, 1)
		        if (forward_appear[0] == 1):
		            forward_temp[0] = 1
		        else:
		            forward_temp[0] = 0        
		        if (forward_appear[1] == 1):
		            forward_temp[1] = 1
		        else:
		            forward_temp[1] = 0
		    elif (forward_road_number == 3):     #3 road
		        forward_road_1 = forward_foreground
	   		forward_road_1 = forward_foreground[forward_road_coord[0, 1] - forward_gaussian_range[1]:forward_road_coord[0, 1] - forward_gaussian_range[1] + forward_road_coord[0, 3],forward_road_coord[0, 0] - forward_gaussian_range[0]:forward_road_coord[0, 0] - forward_gaussian_range[0] + forward_road_coord[0, 2]]

			forward_road_2 = forward_foreground
	   		forward_road_2 = forward_foreground[forward_road_coord[1, 1] - forward_gaussian_range[1]:forward_road_coord[1, 1] - forward_gaussian_range[1] + forward_road_coord[1, 3],forward_road_coord[1, 0] - forward_gaussian_range[0]:forward_road_coord[1, 0] - forward_gaussian_range[0] + forward_road_coord[1, 2]]

			forward_road_3 = forward_foreground
	   		forward_road_3 = forward_foreground[forward_road_coord[2, 1] - forward_gaussian_range[1]:forward_road_coord[2, 1] - forward_gaussian_range[1] + forward_road_coord[2, 3],forward_road_coord[2, 0] - forward_gaussian_range[0]:forward_road_coord[2, 0] - forward_gaussian_range[0] + forward_road_coord[2, 2]]

		        for i_1 in range(0, forward_road_coord[0, 3]):
		            for j_1 in range(0, forward_road_coord[0, 2]):
		                forward_pixel[0] = forward_road_1[i_1, j_1]
		                if (forward_pixel[0] > forward_pixel_threshold):
		                    forward_sum[0] += 1

		        for i_2 in range(0, forward_road_coord[1, 3]):
		            for j_2 in range(0, forward_road_coord[1, 2]):
		                forward_pixel[1] = forward_road_2[i_2, j_2]
		                if (forward_pixel[1] > forward_pixel_threshold):
		                    forward_sum[1] += 1
			
		        for i_3 in range(0, forward_road_coord[2, 3]):
		            for j_3 in range(0, forward_road_coord[2, 2]):
		                forward_pixel[2] = forward_road_3[i_3, j_3]
		                if (forward_pixel[2] > forward_pixel_threshold):
		                    forward_sum[2] += 1

		        if (forward_sum[0] > forward_sum_threshold):
		            forward_appear[0] = 1
			    Sfp_time1 = datetime.datetime.now().strftime('%f')
			    Sp_time1 = datetime.datetime.now().strftime('%S')
		        else:
		            forward_appear[0] = 0
		        if (forward_sum[1] > forward_sum_threshold):
		            forward_appear[1] = 1
			    Sfp_time2 = datetime.datetime.now().strftime('%f')
			    Sp_time2 = datetime.datetime.now().strftime('%S')
		        else:
		            forward_appear[1] = 0
		        if (forward_sum[2] > forward_sum_threshold):
		            forward_appear[2] = 1
			    Sfp_time3 = datetime.datetime.now().strftime('%f')
			    Sp_time3 = datetime.datetime.now().strftime('%S')
		        else:
		            forward_appear[2] = 0
		        if (forward_appear[0] == 0 and forward_temp[0] == 1):
			    YOLO(frame_resized, darknet_image, 0)
		        if (forward_appear[1] == 0 and forward_temp[1] == 1):
		            YOLO(frame_resized, darknet_image, 1)
		        if (forward_appear[2] == 0 and forward_temp[2] == 1):
		            YOLO(frame_resized, darknet_image, 2)
		        if (forward_appear[0] == 1):
		            forward_temp[0] = 1
		        else:
		            forward_temp[0] = 0        
		        if (forward_appear[1] == 1):
		            forward_temp[1] = 1
		        else:
		            forward_temp[1] = 0
		        if (forward_appear[2] == 1):
		            forward_temp[2] = 1
		        else:
		            forward_temp[2] = 0

		for i in range(forward_road_number):   #draw frame
		    frame_resized0=cv2.rectangle(frame_resized0, (count_area[i, 0],count_area[i, 1]),(count_area[i, 0]+count_area[i, 2],count_area[i,1]+count_area[i, 3]), (255, 255, 255), 1)
		image = frame_resized0#detections
		#
		for i in range(forward_road_number + 1):
		    image = cv2.line(image,(int(darknet_road_line[i, 0] * darknet.network_width(netMain)), int(darknet_road_line[i, 1] * darknet.network_height(netMain))), (int(darknet_road_line[i, 2] * darknet.network_width(netMain)), int(darknet_road_line[i, 3] * darknet.network_height(netMain))), (0, 255, 255),1)

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		

		_tmp_str="L" if chk_left(0,208) else "R"
		_tmp_str2="L" if chk_left(400,200) else "R"
		cv2.putText(image,_tmp_str,(0,208),cv2.FONT_HERSHEY_COMPLEX,.5,(255,255,255),1)#L
		cv2.putText(image,_tmp_str2,(400,200),cv2.FONT_HERSHEY_COMPLEX,.5,(255,255,255),1)#R

		cv2.waitKey(3)
		cv2.imshow('Demo', image)
		cv2.waitKey(3)
	    np.savetxt(kname,Csum,delimiter=',',fmt = '%d')
	    cv2.destroyWindow("Demo")
	    cap.release()
	    print("yolo exist")


_Keep_Run=True
#
def YOLO(Cimg, Dimg, Count_Road):
    Sfn_time = int(datetime.datetime.now().strftime('%f'))
    Sn_time = int(datetime.datetime.now().strftime('%S'))

    darknet.copy_image_from_bytes(Dimg,Cimg.tobytes())
    #
    global Sp_time, Sfp_time, Tdis, Sspeed, Bspeed, Rhold, Csum, count
    if (Count_Road == 0):
	Sp_time = int(Sp_time1)
	Sfp_time = int(Sfp_time1)
    elif (Count_Road == 1):
	Sp_time = int(Sp_time2)
	Sfp_time = int(Sfp_time2)
    elif (Count_Road == 2):
	Sp_time = int(Sp_time3)
	Sfp_time = int(Sfp_time3)   
    if (Sn_time < Sp_time):
	Sn_time += 60
    Tdis = ((Sn_time - Sp_time + 0.3) * 1000000 + Sfn_time - Sfp_time) * 0.000001
    
    detections = darknet.detect_image(netMain, metaMain, Dimg, thresh=0.25)

    #print(detections)
    want=[]
    for i in range(len(detections)):	
	if(i >= len(detections)):
	    break                 	
        Mitem = detections[i]
        trust = Mitem[1]
        Ikey = i
    
        for j in range(i + 1, len(detections)):   
	    if(j >= len(detections)):
	        break
	    Compitem = detections[j]
            CompitemX = Compitem[2][0] + Compitem[2][2] / 2
	    CompitemY = Compitem[2][1] + Compitem[2][3] / 2
            if (CompitemX > Mitem[2][0] and CompitemX < Mitem[2][0] + Mitem[2][2] and CompitemY > Mitem[2][1] and CompitemY < Mitem[2][1] + Mitem[2][3]):
                if (Compitem[1] > trust):
                    trust = Mitem[1]
                    detections.remove(detections[Ikey])
                    j -= 1
                    Ikey = j
                    continue
                detections.remove(detections[j])
                j -= 1    
        want.append(detections[Ikey])
    LCP = 0
    Lastcar = 0
    Csum = []
    GCenter = np.array( [ [int(Fgaussian_range[0, 0] * 416 + Fgaussian_range[0, 2] * 416 / 2),int(Fgaussian_range[0, 1] * 416 + Fgaussian_range[0, 3] * 416 / 2)],[int(Fgaussian_range[1, 0] * 416 + Fgaussian_range[1, 2] * 416 / 2),int(Fgaussian_range[1, 1] * 416 + Fgaussian_range[1, 3] * 416 / 2)],[int(Fgaussian_range[2, 0] * 416 + Fgaussian_range[2, 2] * 416 / 2),int(Fgaussian_range[2, 1] * 416 + Fgaussian_range[2, 3] * 416 / 2)] ] )   #gaussain frame center
    LastXY = [0, 0]
    category = [ "sedan", "truck", "scooter", "bus", "Flinkcar", "Hlinkcar"]
    XYrate0 = (float(darknet_road_line[Count_Road, 2] * 416 - darknet_road_line[Count_Road, 0]*416)) / (float(darknet_road_line[Count_Road, 3] *416 - darknet_road_line[Count_Road, 1]*416))
    XYrate1 = (float(darknet_road_line[Count_Road + 1, 2]*416 - darknet_road_line[Count_Road + 1, 0]*416)) / (float(darknet_road_line[Count_Road + 1, 3]*416 - darknet_road_line[Count_Road + 1, 1]*416))

    global count, count_clear, Speed, Vsc, Vbc, Svs, Svb, l_Minsp, l_Maxsp, l_Ct,frame_resized
    if (count_clear == 1):
	Vsc = [0, 0, 0]
        Vbc = [0, 0, 0]
	Bspeed = [0, 0, 0]
	Sspeed = [0, 0, 0]
	Rhold = [0, 0, 0]
	for i in range(3):
	    for j in range(6):
		count[i, j] = 0;
	count_clear = 0
    for detection in want:     #counting
        x, y, w, h = detection[2][0],\
		    detection[2][1],\
		    detection[2][2],\
		    detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
	
	for i in range(len(category)):              
            if (ExistM == 1):
                if (Count_Road == forward_road_number - 1 and detection[0] != category[2]):
                    continue                        
                elif (Count_Road < forward_road_number - 1 and detection[0] == category[2]):                        
                    continue
            if (detection[0] == category[i]):                 
                RLine = int(XYrate0 * (ymax - (darknet_road_line[Count_Road, 1] * 416)) + (darknet_road_line[Count_Road, 0] * 416)) 	    
                LLine = int(XYrate1 * (ymax - (darknet_road_line[Count_Road + 1, 1] * 416)) + (darknet_road_line[Count_Road + 1, 0] * 416)) 
		print('xmax '+str(xmax)+' xmix '+str(xmin)+' Rline '+str(RLine*1.1)+' Lline '+str(LLine/1.3))
                if (xmin - (xmax-xmin)/2  < RLine *1.108  and xmin + (xmax-xmin) > LLine  ):	 #detect frame cneter between right line and left line           
		    Rhold[Count_Road] += (((Sn_time - Sp_time) * 1000000 + Sfn_time - Sfp_time) * 0.000001) * 10	
		    if (LCP == 0):
                        count[Count_Road, i] += 1	

                        LastXY[0] = GCenter[Count_Road, 0] - (x + w / 2)
                        LastXY[1] = GCenter[Count_Road, 1] - (y + h / 2)
                        Lastcar = i
			Speed = (1.3 / (Fgaussian_range[Count_Road, 3] * 416) * 3.6 * h) / Tdis
			if (Speed > l_Minsp and Speed < l_Maxsp):		
			    if (detection[0] == category[0]):
			        Vsc[Count_Road] += 1
			        Sspeed[Count_Road] += Speed
			        Svs[Count_Road] = int(Sspeed[Count_Road] / Vsc[Count_Road])
			    elif (detection[0] == category[1] or detection[0] == category[3] or detection[0] == category[4] or detection[0] == category[5]):
			        Vbc[Count_Road] += 1
			        Bspeed[Count_Road] += Speed
			        Svb[Count_Road] = int(Bspeed[Count_Road] / Vbc[Count_Road])                        
                    else:
                        if (pow(pow(GCenter[Count_Road, 1] - (y + h / 2), 2) + pow(GCenter[Count_Road, 0] - (x + w / 2), 2), 0.5) < pow(pow(LastXY[0], 2) + pow(LastXY[1], 2), 0.5) and (y + h) > GCenter[Count_Road, 1]):
			    Speed = (1.3 / (Fgaussian_range[Count_Road, 3] * 416) * 3.6 * h) / Tdis
			    if (Speed > l_Minsp and Speed < l_Maxsp):
			        if (detection[0] == category[0]):
			            Sspeed[Count_Road] += Speed
				    Vsc[Count_Road] += 1
				    Svs[Count_Road] = int(Sspeed[Count_Road] / Vsc[Count_Road])
			        elif (detection[0] == category[1] or detection[0] == category[3] or detection[0] == category[4] or detection[0] == category[5]):
			            Vbc[Count_Road] += 1
			            Bspeed[Count_Road] += Speed
				    Svb[Count_Road] = int(Bspeed[Count_Road] / Vbc[Count_Road])	    
                            count[Count_Road, Lastcar] -= 1
                            count[Count_Road, i] += 1

                            LastXY[0] = GCenter[Count_Road, 0] - (x + w / 2)
                            LastXY[1] = GCenter[Count_Road, 1] - (y + h / 2)
                            Lastcar = i       
                    LCP += 1 
                break
    Csum = [0, 0, 0, 0, 0, 0]
    
    for i in range(3):
        for j in range(6):
            Csum[j] += count[i, j]
    for i in range(3):
	print("SedanRoad" + str(i) + "   sedan: " + str(count[i, 0]) + " truck: " + str(count[i, 1]) + " scooter: " + str(count[i, 2]) + " bus: " + str(count[i, 3]) + " Flinkcar: " + str(count[i, 4]) + " Hlinkcar: " + str(count[i, 5]))


def catch_exit():
    global _Keep_Run
    print(raw_input())
    _Keep_Run = False

if __name__ == "__main__":

    

    ##all get road and set!
    thread_list=[]
    #init
    init()
    #
    set_area(0,0,1,1)
    set_count_area(0,0,1,1)
    set_road_line(0.1115,0,0.7665,1)#DOWM TO UP
    ExistM = 0
    count = np.array( [ [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0] ] )
    forward_appear = [0, 0, 0, 0]
    forward_temp = [0, 0, 0, 0]
    darknet_road_line = np.array( [ [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0] ] )
    Fgaussian_range = np.array( [ [0.01,0.01,0.01,0.01], [0.01,0.01,0.01,0.01], [0.01,0.01,0.01,0.01] ] )

    forward_road_number = 2

    two_way_frist = 1
    forward_sum_threshold = 100 #200
    forward_pixel_threshold = 125
    forward_pixel = [0, 0, 0]
    forward_pMOG2 = cv2.createBackgroundSubtractorMOG2()
    forward_pMOG3 = cv2.createBackgroundSubtractorMOG2()
    #
    thread_list.append(threading.Thread(target=catch_exit))
    #thread_list.append(threading.Thread(target=tcp_run))
    thread_list.append(threading.Thread(target=GMM))
    for th in thread_list:
        th.start()
    #
    for th in thread_list:
        th.join()





