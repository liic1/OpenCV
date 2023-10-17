from __future__ import print_function
import argparse
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import time
import dlib
from pygame import mixer
import os


# loading alarm sound
mixer.init()
sound = mixer.Sound(f'{os.getcwd()}//old-mechanic-alarm-clock-140410.wav')


# function that calculates the EAR (eye aspect ratio)
#   the very cool mathematical formula using the points of facial landmarking on eyes
#   drops in number when eyes are closed, basically indicating through math that someone has shut their eyes
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


# load in 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


# our general thresholds: 
EYE_AR_THRESH = 0.3 # threshold for when to consider the eyes "closed"; EAR having a value of .3
EYE_AR_CONSEC_FRAMES = 150 # amount of frames that EAR can be below variable above before considered asleep (not sure how long this is tbh)

# initialize the frame counters that determine whether asleep or not
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) 
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"]) # create facial landmark predictor


# grab indexes of the facial landmarks for the left and right eye (they have their own points on the facial landmark map)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
thic = 2 # thickness of the border that flashes when the alarm starts

# looping over frames-- while loop ensures it runs infinitely
# grab the frame from the threaded video file stream, resize
# it, and convert it to grayscale
# channels)
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=900) # resize frame 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
	# detect faces in the grayscale frame
    rects = detector(gray, 0) # using dlib library

    for rect in rects:
		# finds facial landmarks using dlib and puts them in np array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
		# get the left and right eye coordinates & use the
		# coordinates to compute the eye aspect ratio for both eyes (using EAR function we created earlier :D)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes (winks wont count now!)
        ear = (leftEAR + rightEAR) / 2.0
        
        # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
        height,width = frame.shape[:2]
        if ear < EYE_AR_THRESH:
            COUNTER += 1 # counts how many frames the eyes are closed
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "ASLEEP", (350, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.7, (0, 0, 255), 5)
                cv2.putText(frame, "WAKE UP! WAKE UP!", (180, 500), cv2.FONT_HERSHEY_TRIPLEX, 1.7, (0, 0, 255), 5) 
                sound.play()
                if (thic<16):
                    thic += 2
                # make box thinner again, to give it a pulsating appearance
                else:
                    thic -= 2
                    if(thic<2):
                        thic = 2 # set border to 2 and then get rid of it repeatedly
                    cv2.rectangle(frame, (0,0), (width, height), (0,0,255), thickness=thic)
        else:
            COUNTER = 0
            sound.stop()

        # else:
		# 	# if the eyes were closed for a sufficient number of
		# 	# then increment the total number of blinks   
        #     if COUNTER >= EYE_AR_CONSEC_FRAMES:  
        #         TOTAL += 1
		# 	# reset the eye frame counter for the BLINK
        #     # we don't have to reset the counter if the "wake up" is going to stay as long as EAR is below thresh
        #     COUNTER = 0
        # draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
        #cv2.putText(frame, "Time Slept: {}".format(TOTAL), (10, 30),
			#cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        
        
 
	# show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()




# miscelaneous stuff from when I was brainstorming





# # 3: load and display
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())

# image = cv2.imread(args["image"])

# # 6: Resize image
# r = 600.0 / image.shape[0]
# dim = (int(image.shape[1] * r), 600)
# resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("Resized (Height)", resized)
# cv2.waitKey(0)

# # prep facial detection by making image b&w
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", gray)
# cv2.waitKey(0)

# detect face
    # load in face classifier built into CV




# bounding box on video 
"""
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
leye = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")    
reye = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")                      
video_capture = cv2.VideoCapture(0)

# function that will continuously draw bounding box on face
def detect_bounding_box(vid):
    gray_img = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY) #convert image to gray to draw the box better
    personFaces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in personFaces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return personFaces

def detect_eyes_open(vid):
    gray_img = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)


while True: # constantly draw bounding box/detect face

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Video", video_frame
    )  # display the processed frame in a window named "My Face Detection Video"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()                                       
"""


# bounding box on image
"""
    # detectMultiScale identifies faces in an input image
    # scale factor scales image-- I don't want to scale it right now
    # minNeighbors... # of neighboring rectanles that need be identified for a valid detection
    # minSize is the minimum size of face that can be detected
personFace = face_classifier.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

for (x, y, w, h) in personFace:
    print(x, y, w, h)
    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 5)

#image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) # check what you pass into this
cv2.imshow("box", resized)
cv2.waitKey(0)
# plt.figure(figsize=(20,10))
# plt.imshow(image_rgb)
# plt.axis('off')
"""
