import cv2
import time 
import numpy as np


video_path = 'TrackVideo.mov'
output_file = "TrackOutput.mov"     
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,   # How many pts. to locate
                       qualityLevel = 0.1,  # b/w 0 & 1, min. quality below which everyone is rejected
                       minDistance = 7,   # Min eucledian distance b/w corners detected
                       blockSize = 3 ) # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),  # size of the search window at each pyramid level
                  maxLevel = 2,   #  0, pyramids are not used (single level), if set to 1, two levels are used, and so on
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

''' Criteria : Termination criteria for iterative search algorithm.
    after maxcount { Criteria_Count } : no. of max iterations.
    or after { Criteria Epsilon } : search window moves by less than this epsilon '''


# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  #use goodFeaturesToTrack to find the location of the good corner.

# Create a mask image for drawing purposes filed with zeros
mask = np.zeros_like(old_frame)

y = 0
is_begin = True # To save the output video
count = 1  # for the frame count
n = 60  # Frames refresh rate for feature generation

while True:
    ret,frame = cap.read()
    if frame is None:
        break
    processed = frame

    #Saving the Video
    if is_begin:
        h, w, _ = processed.shape
        out = cv2.VideoWriter(output_file, fourcc, 30, (w, h), True)
        is_begin = False

    # Convert to Grey Frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if count%n == 0:  # Refresh the tracking features after every 50 frames
        cv2.imwrite('img/r{0:05d}.jpg'.format(y), img)
        y += 1
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() #tmp new value
        c,d = old.ravel() #tmp old value
        #draws a line connecting the old point with the new point
        mask = cv2.line(mask, (a,b),(c,d), (0,255,0), 1)
        #draws the new point
        frame = cv2.circle(frame,(a,b),2,(0,0,255), -1)
    img = cv2.add(frame,mask)

    out.write(img)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff

    #Show the Output
    if k == 27:
        cv2.imshow('', img)
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    count += 1

# release and destroy all windows

# cap = cv2.VideoCapture('TrackVideo.mov')

# while(cap.isOpened()):
#   ret, frame = cap.read()
#   #time.sleep(0.05) 
#   cv2.imshow('frame',frame)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

cap.release()
cv2.destroyAllWindows()