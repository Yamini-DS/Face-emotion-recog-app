from fastai import *
from fastai.vision import *
#from fastai.vision import image
import os
#import pandas as pd
import numpy as np
import cv2

model4_test=load_learner(path=r"D:\Data science\Alma better\DL Facial emotion recognition\Images\images\train",file='fastai_emojis_model4.pkl')
os.chdir(r'D:\Data science\Alma better\DL Facial emotion recognition\Images\images\Input and output')

#This is the text added to the prediction
def prediction(img1):
    predictions = []
    predictions = model4_test.predict(img1)
    predictions[0]
    # print(predictions)
    # type(predictions)
    prediction1 = []
    prediction1 = str(predictions[0])
    # emotion = []
    # emotion = Emojis_dict[predictions1]
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
    # cv2.destroyWindow("preview")

#the function used to get the predictions from the model
def return_prediction(path):
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

    model_pred1 = model4_test.predict(img1)[0]
    prediction(img1)  # uncomment when above type of display text is required for image outputs
    plt.imshow(img)  # uncomment if image has to be displayed
    return str(model_pred1)

#test to run until we stop or video ends
def test_rerun(text, cap):
    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "The last phase of the person's Emotion was recorded " + str(text), (95, 30), font, 1.0,
                    (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(img, "Press SPACE: Detecting", (5, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(img, "Hold Q: To Quit😎", (460, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite("test7.jpg", img)
            text = return_prediction("test7.jpg")
            test_video_pred(text, cap)
            break

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

#Video detection begins with the below code or function
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('./pexels-tiger-lily-7149007.mp4')

def test_video_pred(text, cap):
    while (True):
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "The last phase of person's emotion was recorded: " + str(text), (95, 30), font, 1.0,
                    (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(img, "Press SPACE: For detection", (5, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(img, "Hold Q: To Quit😎", (460, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite("test7.jpg", img)
            text = return_prediction("test7.jpg")
            test_rerun(text, cap)
            # plt.imshow(img)
            break

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

#cap=cv2.VideoCapture(0)

#test_video_pred('None',cap)