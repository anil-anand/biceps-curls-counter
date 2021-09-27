import mediapipe as mp
import cv2
import numpy as np


def calculate_angle(a, b, c):

    a = np.array(a) #   Change the arrays to numpy arrays
    b = np.array(b) #   a = shoulder point, b = elbow and
    c = np.array(c) #   c = wrist, each of which has two coordinates, x and y

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])  # find the angle in radians
    angle = np.abs(radians*180/np.pi)   # convert the angle in radians to degrees

    if(angle > 180.0):          # convert any angle which is greater than 180 degrees, to less than 180 degrees
        angle = 360 - angle

    return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

leftcounter = 0     # variable to store the curl count of the left hand
leftstage = None    # variable to store the status of the curl: up/down (left hand)

rightcounter = 0     # variable to store the curl count of the right hand
rightstage = None    # variable to store the status of the curl: up/down (right hand)

cap  = cv2.VideoCapture(0)

#Initiate pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while(cap.isOpened()):

        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Recolor input from BGR to RGB for mediapipe processing
        image.flags.writeable = False

        results = pose.process(image)  # Make detections using mediapipe

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Recolor back to BGR from RGB

        try:        # use try-except to make things work, if there's something wrong with the webcam input; no error => calculate angle; error => don't calculate angle, try with next frame

            landmarks = results.pose_landmarks.landmark  # Extract landmarks from the current frame

            leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]  # Take the x and y coordinates of shoulder, elbow
            leftelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]           # and wrist of the left hand to calculate the angle 
            leftwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]           # between the lower and upper arms (left).

            rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]  # Take the x and y coordinates of shoulder, elbow
            rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]           # and wrist of the right hand to calculate the  
            rightwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]           # angle between the lower and upper arms (right).
            
            
            leftangle = calculate_angle(leftshoulder, leftelbow, leftwrist)  # Calculate the angle between the lower and upper arms of the left hand
            rightangle = calculate_angle(rightshoulder, rightelbow, rightwrist)  # Calculate the angle between the lower and upper arms of the right hand

            cv2.putText(image, str(leftangle), tuple(np.multiply(leftelbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)     # Print the angle in the webcam output
            cv2.putText(image, str(rightangle), tuple(np.multiply(rightelbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)     # Print the angle in the webcam output



            if(leftangle > 160):                            #  Calculate/update
                leftstage = 'Down'                          #  the values of
            elif(leftangle < 45 and leftstage == 'Down'):   #  counter and stage
                leftcounter += 1                            #  variables of the left
                leftstage = ' Up'                           #  hand for each frame

            if(rightangle > 160):                            #  Calculate/update
                rightstage = 'Down'                          #  the values of
            elif(rightangle < 45 and rightstage == 'Down'):  #  counter and stage
                rightcounter += 1                            #  variables of the right
                rightstage = ' Up'                           #  hand for each frame




            cv2.rectangle(image, (0, 0), (225, 73), (225,95, 50), -1)   # Plot a rectangle at the top left corner

            cv2.rectangle(image, (415, 0), (640, 73), (225,95, 50), -1) # Plot a rectangle at the top right corner



            cv2.putText(image, 'Left Hand', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA) # Text headings
            cv2.putText(image, 'Curls', (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)     # in the left
            cv2.putText(image, 'Stage', (160, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)    # hand box


            cv2.putText(image, str(leftcounter), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 2, cv2.LINE_AA) # Curl count and stage
            cv2.putText(image, leftstage, (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)          # for the left hand

            
            
            cv2.putText(image, 'Right Hand', (480, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA) # Text headings
            cv2.putText(image, 'Curls', (425, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)      # in the right
            cv2.putText(image, 'Stage', (570, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)      # hand box
           

            cv2.putText(image, str(rightcounter), (430, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 2, cv2.LINE_AA) # Curl count and stage
            cv2.putText(image, rightstage, (550, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)           # for the right hand
           

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # Draw pose landmarks on output

        cv2.imshow('Raw webcam feed', image) # Output camera feed through opencv

        if(cv2.waitKey(10) & 0xFF == ord('q')): # Exit when the key 'q' is pressed on the keyboard
            break

cv2.release()
cv2.destroyAllWindows()