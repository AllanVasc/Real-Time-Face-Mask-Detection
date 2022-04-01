# All necessary imports

import numpy as np
import cv2
from keras.models import load_model
import time

# Loading Face Detector

cameraW = 640
cameraH = 480

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# For better face detection, at the expense of FPS rate, use the other models present in the repository.

threshold = 0.9
camera = cv2.VideoCapture(0)
camera.set(3, cameraW)
camera.set(4, cameraH)
text_font = cv2.FONT_HERSHEY_COMPLEX

# Loading Model

imgDimension = (128,128,3)
model = load_model('model')

# Preprocessing function

def preprocessing_img(img):
  img = img/255.0
  return img

# Get the name of the class to show in application

def getClassName(prediction):
  if prediction == 0:
    return "No Mask"
  elif prediction == 1:
    return 'Mask'

# Our application

pTime = 0

while True:

  sucess, imgOriginal = camera.read()
  faces = faceDetector.detectMultiScale(imgOriginal, 1.3, 5)

  for x,y,w,h in faces:

    print("Detected Faces...")
    cropped_img = imgOriginal[y:y+h, x:x+h] # taking the region only of the face

    # Preprocessing img

    img = cv2.resize(cropped_img, (imgDimension[0], imgDimension[1]))
    img = preprocessing_img(img)
    img = img.reshape(1, imgDimension[0],imgDimension[1], imgDimension[2]) # Make this for inference !

    # Running Model

    prediction = model.predict(img)
    classIndex = np.argmax(prediction)
    probabilityValue = np.amax(prediction)

    print('Class Index = ', classIndex)
    print('probabilityValue = ', probabilityValue)

    if probabilityValue >= threshold:
      if classIndex == 0: # Without Mask
        cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(imgOriginal, (x,y-40),(x+w, y), (50,50,255),-2)
        cv2.putText(imgOriginal, str(getClassName(classIndex)),(x,y-10), text_font, 0.75, (255,255,255),1, cv2.LINE_AA)
      elif classIndex == 1: # With mask
        cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(imgOriginal, (x,y-40),(x+w, y), (0,255,0),-2)
        cv2.putText(imgOriginal, str(getClassName(classIndex)),(x,y-10), text_font, 0.75, (255,255,255),1, cv2.LINE_AA)
	
  # Showing image and FPS

  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime
  cv2.putText(imgOriginal, f'FPS: {int(fps)}', (20,30), text_font, 1, (255,0,0), 2)
  cv2.putText(imgOriginal, f'Press "q" for exit...', (20,cameraH-20), text_font, 1, (0,0,255), 2)
  cv2.imshow("Face Mask Detector",imgOriginal)

  # Wait for the key to finish

  k = cv2.waitKey(1)
  if k == ord('q'):
    break

# End application

camera.release()
cv2.destroyAllWindows()