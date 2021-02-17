# import the necessary packages
from imutils.video import VideoStream
from pyzbar import pyzbar
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="qrcodes.csv",
	help="path to output CSV file containing qrcodes")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# open the output CSV file for writing and initialize the set of
# qrs found thus far
csv = open(args["output"], "w")
found = set()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it to
	# have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# find the qrs in the frame and decode each of the qrs
	qrs = pyzbar.decode(frame)

    # loop over the detected qrs
	for qr in qrs:
		# extract the bounding box location of the qr and draw
		# the bounding box surrounding the qr on the image
		(x, y, w, h) = qr.rect
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
		# the qr data is a bytes object so if we want to draw it
		# on our output image we need to convert it to a string first
		qrData = qr.data.decode("utf-8")
		qrType = qr.type
		# draw the qr data and qr type on the image
		text = "{} ({})".format(qrData, qrType)
		cv2.putText(frame, text, (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		# if the qr text is currently not in our CSV file, write
		# the timestamp + qr to disk and update the set
		if qrData not in found:
			csv.write("{},{}\n".format(datetime.datetime.now(),
				qrData))
			csv.flush()
			found.add(qrData)
    # show the output frame
	cv2.imshow("qr Scanner", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# close the output CSV file do a bit of cleanup
print("[INFO] cleaning up...")
csv.close()
cv2.destroyAllWindows()
vs.stop()