# USAGE yoo
# python3 webstreaming.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --output output/output_01.avi --ip 0.0.0.0 --port 8000

# import the necessary packages
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


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
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

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def people_counter(prototxt,model,output,confidences,skipframes):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
	ids = (1,)

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0

	# loop over frames from the video stream
	while True:
		frame = vs.read()
        frame = frame[1]
        frame = imutils.resize(frame, width = 400)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            
        if output is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W,H), True)
        
        rects = []
        
        if totalFrames % skipframes == 0:
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0,0,i,2]
                if confidence > confidences:
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
            #datacctv.append({'timestamp' : time, 'tipe' : tipe, 'file foto' : filefoto})
        totalFrames += 1
        with lock:
            outputFrame = frame.copy()
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-c", "--confidence", type=float, default=0.51, help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=2, help="# of skip frames between detections")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=people_counter, args=(args["prototxt"],args["model"],args["output"], args["confidence"], args["skip-frames"]))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
