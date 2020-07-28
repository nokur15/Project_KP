# file
# python3 people_counter_tab.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 --output output/output_01.avi
# webcam
# python3 people_counter_tab.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --output output/output_01.avi

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import pandas as pd
import argparse
import imutils
import time
import dlib
import cv2
import csv
import mysql.connector
from datetime import datetime

def copyf(dictlist, key, valuelist):
    return [dictio for dictio in dictlist if dictio[key] in valuelist]
    
def isempty(x):
    if not x: return True
    
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
        return True
    else:
        return False

def filefotobeda(x,y):
    same = 0
    for a in y:
        if x == a:
            same += 1
    if same > 0:
        return False
    else:
        return True

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.51, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=2, help="# of skip frames between detections")
args = vars(ap.parse_args())

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="serenaochacohina",
  database="datacctv"
)

mycursor = mydb.cursor()
mycursor.execute("CREATE TABLE data (id INT AUTO_INCREMENT PRIMARY KEY, timestamp VARCHAR(255), tipe VARCHAR(255), file_foto VARCHAR(255))")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])
    
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

fps = FPS().start()

while True:
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame
    if args["input"] is not None and frame is None:
        break
    frame = imutils.resize(frame, width = 400)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W,H), True)
    
    rects = []
    
    if totalFrames % args["skip_frames"] == 0:
        trackers = []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > args["confidence"]:
                idx = int(detections[0,0,i,1])
                if classfilter(CLASSES[idx]):
                    continue
                box = detections[0,0,i,3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                trackers.append(tracker)
    else:
        for tracker in trackers:
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            rects.append((startX, startY, endX, endY))
    
    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
        trackableObjects[objectID] = to
        text = "ID {}".format(objectID)
        cv2.imwrite("data/user."+str(objectID)+".jpg", frame)
        filefoto = "data/user."+str(objectID)+".jpg"
        time = datetime.now()
        tipe = CLASSES[idx]
        sql = "SELECT file_foto FROM data WHERE id IN (%s)" % (', '.join(str(id) for id in ids))
        mycursor.execute(sql)
        comp = mycursor.fetchall()
        filefotocomp = ("data/user."+str(objectID)+".jpg",)
        if filefotobeda(filefotocomp,comp):
            sql = "INSERT INTO data (timestamp, tipe, file_foto) VALUES (%s, %s, %s)"
            val =(time,tipe,filefoto)
            mycursor.execute(sql,val)
            mydb.commit()
            sql = "SELECT id FROM data WHERE file_foto = %s"
            val = (filefoto,)
            mycursor.execute(sql,val)
            inp_id = mycursor.fetchall()
            for x in inp_id:
                ids = ids + x
        datacctv.append({'timestamp' : time, 'tipe' : tipe, 'file foto' : filefoto})
    
    if writer is not None:
        writer.write(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
    totalFrames += 1
    fps.update()
    
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#print(ids)
#print(inp_id)

i = 0
data_cctv_fin = []
while True:
    filebase = ['data/user.'+str(i)+'.jpg']
    x = copyf(datacctv, 'file foto', filebase)
    if isempty(x):
        break
    else:
        data_cctv_fin.append(x[0])
        i+=1

fields = ['timestamp', 'tipe', 'file foto']
filename = "data_cctv.csv"
with open(filename, 'w') as csvfile:
    writercsv = csv.DictWriter(csvfile, fieldnames = fields)
    writercsv.writeheader()
    writercsv.writerows(data_cctv_fin)
    
if writer is not None:
    writer.release()

if not args.get("input", False):
    vs.stop()
else:
    vs.release()
    
cv2.destroyAllWindows()

