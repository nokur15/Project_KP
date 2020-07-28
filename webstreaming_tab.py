# python3 webstreaming_tab.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --output output/output_01.avi --ip 0.0.0.0 --port 8000

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import numpy as np
import pandas as pd
import threading
import argparse
import datetime
import imutils
import time
import cv2
import dlib
import mysql.connector
from datetime import datetime

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="serenaochacohina",
    database="datacctv"
)

mycursor = mydb.cursor()
mycursor.execute("CREATE TABLE data (id INT AUTO_INCREMENT PRIMARY KEY, timestamp VARCHAR(255), tipe VARCHAR(255), file_foto VARCHAR(255))")
outputFrame = None
lock = threading.lock()

app = Flask(__name__)

print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

writer = None
datacctv = []
ids = (1,)

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0

def classfilter(x):
    filt = 1
    if x == "person":
        filt = 0
    if x == "car":
        filt = 0
    if x == "cat":
        filt = 0
    if x == "motorbike":
        filt = 0
    if x == "dog":
        filt = 0
    if filt == 1:
        return False
    else:
        return True

def filefotobeda(x,y):
    same = 0
    for a in y:
        if x == a:
            same += 1
    if same > 0:
        return False
    else:
    return True
    
@app.route("/")
def index():
    return render_template("index.html")

def people_counter(prototxt,model,output,confidences,skipframes):
    global vs, outputFrame, lock
    
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0,255,size=(len(CLASSES), 3))
    
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt,model)
    ids = (1,)
    
    W = None
    H = None
    
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects={}
    
    totalFrames = 0
    
    while True:
        frame = vs.read()
        frame = frame[1]
        frame = imutils.resize(frame, width = 400)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if W is None or H is None:
            (H,W) = frame.shape[:2]
        
        if
