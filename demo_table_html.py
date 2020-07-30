from flask import Flask, render_template
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
	htable=u'<table width ="70%" border="1" bordercolor=000000 cellspacing="0" cellpadding="1" style="table-layout:fixed;vertical-align:bottom;font-size:13px;font-family:verdana,sans,sans-serif;border-collapse:collapse;border:1px solid rgb(130,130,130)" >'
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

query = "SELECT * FROM data"
print(nlist_to_html(query_mysql(query)))
