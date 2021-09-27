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

counter = 0     # variable to store the curl count
stage = None    # variable to store the status of the curl: up/down

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

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]  # Take the x and y coordinates of shoulder,
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]           # elbow and wrist to calculate the angle 
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]           # between the lower and upper arms.

            angle = calculate_angle(shoulder, elbow, wrist)  # Calculate the angle between the lower and upper arms

            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)     # Print the angle in the webcam output

            if(angle > 160):                            #  Calculate/update
                stage = 'Down'                          #  the values of
            elif(angle < 30 and stage == 'Down'):       #  counter and
                counter += 1                            #  stage variables
                stage = 'Up'                            #  for each frame

            cv2.rectangle(image, (0, 0), (225, 73), (225,95, 50), -1)   # Plot a rectangle at the top left corner

            cv2.rectangle(image, (415, 0), (640, 73), (225,95, 50), -1) # Plot a rectangle at the top right corner

            cv2.putText(image, 'Curls', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)           # Print the number
                                                                                                                    # of curls in the 
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)  # left box

            
            
            cv2.putText(image, 'Stage', (430, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)      # Print the hand
                                                                                                                # up/down status in
            cv2.putText(image, stage, (425, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)    # the right box



        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # Draw pose landmarks on output

        cv2.imshow('Raw webcam feed', image) # Output camera feed through opencv

        if(cv2.waitKey(10) & 0xFF == ord('q')):
            break

cv2.release()
cv2.destroyAllWindows()