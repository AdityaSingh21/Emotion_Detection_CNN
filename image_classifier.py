from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
images = Path("Celeb_Photos").glob("*.jpg")
actors = [str(p) for p in images]
data = list()

def Sentiment_classifier(n,img):
    classifier =load_model('Emotion_little_vgg.h5')
    
    class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the input image
    
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # Display the output
    try:
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
        else:
            return("No face found")
        preds = classifier.predict(roi)[0]
        label=class_labels[preds.argmax()]
        return label
    except:
        return("No face found")
        
        

def Setiment_to_csv():
    for n in actors:
        img = cv2.imread(n)
        sentiment = Sentiment_classifier(n,img)
        data.append([n,sentiment])
    for n in data:
        n[0]=n[0][13:-4]
    df1 =pd.DataFrame(data,columns = ['Actor Name', 'Sentiment'])
    df1.to_csv('Actor_Image_Sentiment.csv')
    
    
if __name__ == "__main__":
    Setiment_to_csv()
        
    
    
    
    
    