from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import streamlit as st
from fastai import *
from fastai.vision import *
import cv2


@st.cache(allow_output_mutation=True)
def get_model():
    model = load_learner(path=r"D:\Data science\Alma better\DL Facial emotion recognition\Images\images\train",
                         file='fastai_emojis_model4.pkl')
    print('Model Loaded')
    return model


def return_prediction(path):
    loaded_model = get_model()
    # converting image to gray scale and save it
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, gray)

    # detect face in image, crop it then resize it then save it
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_clip = img[y:y + h, x:x + w]
        cv2.imwrite(path, cv2.resize(face_clip, (350, 350)))

    # read the processed image then make prediction and display the result
    read_image = cv2.imread(path)
    t = pil2tensor(read_image, dtype=np.float32)  # converts to numpy tensor
    t = t.float() / 255.0
    # t = t.permute((2,0,1))
    # t=t.transpose((2,0,1))

    img1 = Image(t)  # Convert to fastAi Image - this class has "apply_tfms"

    model_pred1 = loaded_model.predict(img1)[0]
    # prediction(img1)  # uncomment when above type of display text is required for image outputs
    plt.imshow(img)  # uncomment if image has to be displayed
    return str(model_pred1)


def prediction(img1):
    predictions = []
    loaded_model = get_model()
    predictions = loaded_model.predict(img1)
    predictions[0]
    prediction1 = []
    prediction1 = str(predictions[0])
    if prediction1 == 'angry':
        print("The person here is angry")
    elif prediction1 == 'disgust':
        print("The person here is disgusted")
    elif prediction1 == 'fear':
        print("The person here is in fear")
    elif prediction1 == 'happy':
        print("The person here is happy")
    elif prediction1 == 'neutral':
        print("The person here is neutral")
    elif prediction1 == 'sad':
        print("The person here is sad")
    elif prediction1 == 'surprise':
        print("The person here is surprised")
    else:
        print("Cannot detect")
