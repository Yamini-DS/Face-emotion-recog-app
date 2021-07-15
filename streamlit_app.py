from typing import List, Any
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from click import style
from fastai import *
from fastai.vision import *
import io
import streamlit.components.v1 as components
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import classify

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1551523713-c1473aa01d9f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=701&q=80")
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""<style>body{
[theme]
backgroundColor="#9fb2d8"
secondaryBackgroundColor="#6072e0"
textColor="#e807bc"
}</style>""", unsafe_allow_html=True)

st.title('Welcome to Face Emotion Recognition Application')

option = st.radio('Which type of detection would you like to make?', ('an Image', 'a Video', 'a Live','testing live'))
st.header('You selected {} option for emotion detection'.format(option))

if option == 'an Image':

    uploaded_file = st.file_uploader("Choose an image", type=['jpg'])
    if uploaded_file is not None:
        # image2 = Image.open(uploaded_file)
        # st.write('Image 2')
        # st.write(type(image2))
        # st.image(image2, caption='Uploaded Image', use_column_width=True)
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR", use_column_width=True)
        # st.write(type(opencv_image))
        gray1 = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        # st.write(type(gray1))

        if st.button('Detect the Emotion'):
            st.write("Result...")
            st.write("Model Loaded")
            model4 = classify.get_model()
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            # read the processed image then make prediction and display the result

            t = pil2tensor(opencv_image, dtype=np.float32)  # converts to numpy tensor
            t = t.float() / 255.0
            img1 = Image(t)

            # Convert to fastAi Image - this class has "apply_tfms"

            model_pred1 = model4.predict(img1)[0]

            st.write(str(model_pred1))
            model_pred2 = classify.prediction(img1)

            st.subheader('Detection made by fastai model using streamlit: {}'.format(str(model_pred1).capitalize()))

elif option == 'a Video':
    st.subheader("Play the Uploaded File while detecting")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    temporary_location = False

    if uploaded_file is not None:
        g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
        temporary_location = "testout_simple.mp4"

        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file

        # close file
        out.close()


    @st.cache(allow_output_mutation=True)
    # @st.cache(suppress_st_warning=True)
    def get_cap(location):
        print("Loading in function", str(location))
        video_stream = cv2.VideoCapture(str(location))

        # Check if camera opened successfully
        if video_stream.isOpened() == False:
            print("Error opening video file")
        return video_stream


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    scaling_factorx = 0.25
    scaling_factory = 0.25
    image_placeholder = st.empty()


    # test to run until we stop or video ends
    def test_rerun(text, video_stream):
        while True:
            ret, image = video_stream.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "The last phase of the person's Emotion was recorded " + str(text), (95, 30), font, 1.0,
                        (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(image, "Press SPACE: Detecting", (5, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(image, "Hold Q: To Quit😎", (460, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Image", image)

            if cv2.waitKey(1) == ord(' '):
                cv2.imwrite("test7.jpg", image)
                model5 = classify.get_model()
                # st.write('Model Loaded')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # read the processed image then make prediction and display the result
                t = pil2tensor(image, dtype=np.float32)  # converts to numpy tensor
                t = t.float() / 255.0
                img1 = Image(t)
                text = model5.predict(img1)[0]
                text = str(text)
                # text
                print(text)
                # st.write(text)
                test_video_pred(text)
                break

            if cv2.waitKey(1) == ord('q'):
                video_stream.release()
                cv2.destroyAllWindows()
                break
        return text


    @st.cache(allow_output_mutation=True)
    # @st.cache(suppress_st_warning=True)
    def test_video_pred(text):
        if temporary_location:
            while True:
                # here it is a CV2 object
                video_stream = get_cap(temporary_location)
                # video_stream = video_stream.read()
                ret, image = video_stream.read()
                if ret:
                    image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory,
                                       interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, "The last phase of person's emotion was recorded: " + str(text), (95, 30), font,
                                1.0,
                                (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, "Press SPACE: For detection", (5, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, "Hold Q: To Quit😎", (460, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for x, y, w, h in faces:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    cv2.imshow("Image", image)

                    if cv2.waitKey(1) == ord(' '):
                        cv2.imwrite("test7.jpg", image)
                        model5 = classify.get_model()
                        # st.write('Model Loaded from test_video_pred')
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        # read the processed image then make prediction and display the result

                        t = pil2tensor(image, dtype=np.float32)  # converts to numpy tensor
                        t = t.float() / 255.0
                        img1 = Image(t)
                        text = model5.predict(img1)[0]
                        text = str(text)
                        # st.write(text)
                        # text
                        print(text)
                        # st.write(str(text))
                        test_rerun(text, video_stream)
                        # plt.imshow(img)
                        break

                    if cv2.waitKey(1) == ord('q'):
                        video_stream.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    print("there was a problem or video was finished")
                    cv2.destroyAllWindows()
                    video_stream.release()
                    break
                # check if frame is None
                if image is None:
                    print("there was a problem None")
                    # if True break the infinite loop
                    break

                image_placeholder.image(image, channels="BGR", use_column_width=True)

                cv2.destroyAllWindows()
            video_stream.release()

            cv2.destroyAllWindows()
        return text


    test_video_pred('None')


elif option == 'a Live':

    st.subheader('Play the Live video while detecting')


    @st.cache(allow_output_mutation=True)
    # @st.cache(suppress_st_warning=True)
    def get_cap_live():
        print("Loading in function")
        video_stream = cv2.VideoCapture(0)

        # Check if camera opened successfully
        if video_stream.isOpened() == False:
            print("Error opening video file")
        return video_stream


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    scaling_factorx = 0.25
    scaling_factory = 0.25
    image_placeholder = st.empty()

    # test to run until we stop or video ends
    def test_rerun(text, video_stream):
        while True:
            ret, image = video_stream.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "The last phase of the person's Emotion was recorded " + str(text), (95, 30), font, 1.0,
                        (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(image, "Press SPACE: Detecting", (5, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(image, "Hold Q: To Quit😎", (460, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Image", image)

            if cv2.waitKey(1) == ord(' '):
                cv2.imwrite("test8.jpg", image)
                model5 = classify.get_model()
                # st.write('Model Loaded')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # read the processed image then make prediction and display the result
                t = pil2tensor(image, dtype=np.float32)  # converts to numpy tensor
                t = t.float() / 255.0
                img1 = Image(t)
                text = model5.predict(img1)[0]
                text = str(text)
                # text
                print(text)
                # st.write(text)
                test_video_pred(text)
                break

            if cv2.waitKey(1) == ord('q'):
                video_stream.release()
                cv2.destroyAllWindows()
                break
        return text


    @st.cache(allow_output_mutation=True)
    # @st.cache(suppress_st_warning=True)
    def test_video_pred(text):
        while True:
            # here it is a CV2 object
            video_stream = get_cap_live()
            # video_stream = video_stream.read()
            ret, image = video_stream.read()
            if ret:
                image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, "The last phase of person's emotion was recorded: " + str(text), (95, 30), font, 1.0,
                            (255, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(image, "Press SPACE: For detection", (5, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(image, "Hold Q: To Quit😎", (460, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for x, y, w, h in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.imshow("Image", image)

                if cv2.waitKey(1) == ord(' '):
                    cv2.imwrite("test8.jpg", image)
                    model5 = classify.get_model()
                    # st.write('Model Loaded from test_video_pred')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # read the processed image then make prediction and display the result

                    t = pil2tensor(image, dtype=np.float32)  # converts to numpy tensor
                    t = t.float() / 255.0
                    img1 = Image(t)
                    text = model5.predict(img1)[0]
                    text = str(text)
                    # st.write(text)
                    # text
                    print(text)
                    # st.write(str(text))
                    test_rerun(text, video_stream)
                    # plt.imshow(img)
                    break

                if cv2.waitKey(1) == ord('q'):
                    video_stream.release()
                    cv2.destroyAllWindows()
                    break
            else:
                print("there was a problem or video was finished")
                cv2.destroyAllWindows()
                video_stream.release()
                break
            # check if frame is None
            if image is None:
                print("there was a problem None")
                # if True break the infinite loop
                break

            image_placeholder.image(image, channels="BGR", use_column_width=True)

            cv2.destroyAllWindows()
            video_stream.release()

            cv2.destroyAllWindows()
        return text


    test_video_pred('None')

elif option == 'testing live':

    webrtc_streamer(key="example")
    st.write('Live functioning')
    st.write('This is running using newly introduced webrtc tool which can access the camera whereas opencv cannot function properly in streamlit then still it has some problems as on screen')
    st.write('This is the end of the testing live notes. See you')

else:
    st.write('You did not select the proper option as specified. Please select a valid option')
    st.write('If one of the four options was selected and it did not work. Please clear the cache and rerun the application')
    st.write('Thanks for understanding')

st.write('Thank you. I hope you got emotions detected which are hidden in the picture or an image or a video')
st.write('See you soon')
st.write('This is created by Yamini')