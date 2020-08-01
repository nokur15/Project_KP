# python3 webserver.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel

from flask import Flask, request, url_for, redirect
from flask_socketio import SocketIO,send,emit
from flask import render_template
import cv2
import base64
from threading import Thread
from time import sleep
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
import csv
import mysql.connector
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.51, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=2, help="# of skip frames between detections")
args = vars(ap.parse_args())

def call_at_interval(period, callback, args):
    while True:
        sleep(period)
        callback(*args)

def setInterval(period, callback, *args):
    Thread(target=call_at_interval, args=(period, callback, args)).start()


def query_mysql(query):
	cnx = mysql.connector.connect(user='root', password='serenaochacohina',
								  host='localhost',
								  database='datacctv',charset="utf8", use_unicode = True)
	cursor = cnx.cursor()
	cursor.execute(query)
	#get header and rows
	header = [i[0] for i in cursor.description]
	rows = [list(i) for i in cursor.fetchall()]
	#append header to rows
	rows.insert(0,header)
	cursor.close()
	cnx.close()
	return rows

#take list of lists as argument
def nlist_to_html(list2d):
	#bold header
	htable=u'<table width ="70%">'
	list2d[0] = [u'<b>' + i + u'</b>' for i in list2d[0]]
	for row in list2d:
		newrow = u'<tr>'
		newrow += u'<td align="left" style="padding:1px 4px">'+str(row[0])+u'</td>'
		row.remove(row[0])
		newrow = newrow + ''.join([u'<td align="right" style="padding:1px 4px">' + str(x) + u'</td>' for x in row])
		newrow += '</tr>'
		htable+= newrow
	htable += '</table>'
	return htable

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

W = None
H = None
datacctv = []
ids = (1,)
idk = 0

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0

# wCap = cv2.VideoCapture('http://192.168.1.3:4747/video')
wCap = cv2.VideoCapture(1)
wCap.set(cv2.CAP_PROP_FRAME_WIDTH,600)
wCap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
camstat = "OFF"
FPS = 20

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/gambar')
def gambar():
    retval, frame = wCap.read()
    retval, buffer = cv2.imencode('.jpg', frame)
    data = base64.b64encode(buffer)
    # wCap.release()
    return data

@app.route('/tabel')
def tabel():
    #global mycursor
    query = "SELECT * FROM data"
    hasil = nlist_to_html(query_mysql(query))
    #mycursor.execute(query)
    #datasini = mycursor.fetchall()
    return hasil




def hello(word):
    global W, H, ct, totalFrames, trackers, trackableObjects, net, CLASSES, COLORS, mycursor, datacctv, ids, idk
    retval, frame = wCap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
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
                idk = idx
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
        tipe = CLASSES[idk]
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
    totalFrames += 1
    retval, buffer = cv2.imencode('.jpg', frame)
    data = base64.b64encode(buffer)
    socketio.emit('kirei', data)

# def cameraOff(arg):
#     print('camera mati')
#     wCap.release()

setInterval(1/FPS, hello, '')





if __name__ == '__main__':
    socketio.run(app)
