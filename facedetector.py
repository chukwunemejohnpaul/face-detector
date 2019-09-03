import cv2
import numpy as np
import os


video = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromCaffe(os.getcwd()+"/deploy.prototxt.txt",os.getcwd()+"/res10_300x300_ssd_iter_140000.caffemodel")

while True:
	ret, Frame = video.read()
	if ret:
		(h,w) = Frame.shape[:2]
		blob  = cv2.dnn.blobFromImage(cv2.resize(Frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
		net.setInput(blob)
		detections = net.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0,0,i,2]
			if confidence > 0.8:
				box  = detections[0,0,i,3:7] * np.array([w,h,w,h])
				(startx, starty, endx, endy) = box.astype("int")
				face = Frame[starty:endy,startx:endx]
				cv2.rectangle(Frame, (startx, starty), (endx, endy),(0, 0, 255), 2)
				cv2.putText(Frame,"this is a face", (startx, starty),cv2.FONT_HERSHEY_SIMPLEX, 1.00, (0, 0, 255), 2)
		cv2.imshow("video",Frame)
		k = cv2.waitKey(1)
		if k == ord("q") & 0xff:
			break
video.release()
cv2.destroyAllWindows()