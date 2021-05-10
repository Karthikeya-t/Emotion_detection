from numpy.testing._private.utils import suppress_warnings
import streamlit as st


from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as SImage
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tempfile


DEMO_IMAGE = 'itony.jpg'

classifier = load_model('model_emo.h5')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class_labels=['anger','disgust', 'sad','happiness', 'surprise']


@st.cache
def detect_emotion(image):
    #resize the frame to process it quickly
    frame = image
   
    
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(96,96),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)

            preds=classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),2)
    
    
    
    return frame
    
                             


st.title('Emotion Detection')


img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))
    

st.subheader('Original Image')

st.image(image, caption=f"Original Image",use_column_width= True)

emotion_analysis = detect_emotion(image)


st.subheader('Emotion Analysis')

st.image(emotion_analysis, caption=f"detected Image",use_column_width= True)




