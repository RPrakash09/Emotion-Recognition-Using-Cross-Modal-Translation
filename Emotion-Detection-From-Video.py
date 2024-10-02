import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import glob
from playsound import playsound
emotion_model = model_from_json(open("fer.json", "r").read())
emotion_model.load_weights('fer.h5')
import time

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cv2.namedWindow('Facial Emotion Prediction')
predicted_emotion = ""
cap=cv2.VideoCapture(0)
my_wait = False
while True:
    ret, img_captured = cap.read()
    if not ret or predicted_emotion != "":
        if predicted_emotion != "":
            playsound(f'{mp3_list[r_int]}')
            predicted_emotion = ""
        continue
    cvt_gray_image = cv2.cvtColor(img_captured, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(cvt_gray_image, 1.32, 5)
    

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(img_captured,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=cvt_gray_image[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = emotion_model.predict(img_pixels)

        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        mp3_list = glob.glob(f'music/{predicted_emotion}*.mp3')
        r_int = np.random.randint(0, len(mp3_list))
        cv2.putText(img_captured, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        #time.sleep(10)
        
    resized_img = cv2.resize(img_captured, (1000, 700))
    cv2.imshow('Facial Emotion Prediction',resized_img)


    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()