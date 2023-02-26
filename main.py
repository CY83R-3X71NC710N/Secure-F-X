
#!/usr/bin/env python
# CY83R-3X71NC710N Copyright 2023

# Secure-F-X is a Python-based facial recognition system designed to detect threats and respond accordingly using openCV, dlib, PIL libraries and ML algorithms tracking motion and facial recognition data to block cyber intrusions.

# Importing necessary libraries
import cv2
import dlib
import PIL
import numpy as np

# Setting up the camera
camera = cv2.VideoCapture(0)

# Setting up the facial recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Setting up the motion detection model
fgbg = cv2.createBackgroundSubtractorMOG2()

# Setting up the facial recognition parameters
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Setting up the facial recognition database
face_database = np.load('face_database.npy').reshape(20, 50*50*3)

# Setting up the facial recognition labels
labels = np.load('labels.npy')

# Setting up the facial recognition model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(face_database, labels)

# Setting up the facial recognition threshold
threshold = 100

# Setting up the facial recognition response
response = 'Access Denied'

# Setting up the facial recognition loop
while True:
    # Capturing the frame
    _, frame = camera.read()

    # Detecting motion
    fgmask = fgbg.apply(frame)

    # Detecting faces
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # Looping through the faces
    for (x, y, w, h) in faces:
        # Detecting facial landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(gray, dlib.rectangle(x, y, x+w, y+h))

        # Extracting the face
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50))

        # Predicting the face
        label, confidence = model.predict(face.reshape(1, 50*50*3))

        # Checking the confidence
        if confidence < threshold:
            response = 'Access Granted'
        else:
            response = 'Access Denied'

        # Drawing the rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Drawing the response
        cv2.putText(frame, response, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Showing the frame
    cv2.imshow('Secure-F-X', frame)

    # Breaking the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the camera
camera.release()

# Destroying all windows
cv2.destroyAllWindows()
