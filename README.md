# Face-emotion-recog-app
- This application is created to find the human emotions anytime and bring human interactions much better and also in solving the problem in education industry so that teachers know the student emotion and keep an eye on them and bring more innovative ways to bring a revolution in education. This helps in bringing more literates which automatically changes the economy, life style and development of the country.
- Face emotion recognition is the process of detecting human emotions from facial expressions which helps in understanding the people better. There are a lot of research going on this topic so that the interactions of humans become more meaningful and has a lot of benefits with it.
- This type of system can be userful for any industry. 
- This can be used in medical field for detection in solving the probelms who are facing pyschological/mental disorders, for safety and security purposes by different controlling authorities, education field for surveillance of students behaviour by educating them in proper way, in technological fields for robot-human interactions, and many more.
The objective behind creating this project is for solving the problem within education and training industry
### Problem statement
The Indian education landscape has been undergoing rapid changes for the past 10 years owing to
the advancement of web-based learning services, specifically, eLearning platforms.
Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India
is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market
is growing on a rapid scale, there are major challenges associated with digital learning when
compared with brick and mortar classrooms. One of many challenges is how to ensure quality
learning for students. Digital platforms might overpower physical classrooms in terms of content
quality but when it comes to understanding whether students are able to grasp the content in a live
class scenario is yet an open-end challenge.
In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the
class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who
need special attention. Digital classrooms are conducted via video telephony software program (exZoom) where it’s not possible for medium scale class (25-50) to see all students and access the
mood. Because of this drawback, students are not focusing on content due to lack of surveillance.
While digital platforms have limitations in terms of physical surveillance but it comes with the power of
data and machines which can work for you. It provides data in the form of video, audio, and texts
which can be analysed using deep learning algorithms. Deep learning backed system not only solves
the surveillance issue, but it also removes the human bias from the system, and all information is no
longer in the teacher’s brain rather translated in numbers that can be analysed and tracked.
### Dataset
The data is collected from kaggle 
Link: https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset
### Model 
The model I used for solving this problem is through resnet34 from fastai transfer learning method.
Even though fastai is recent technology and uses pytorch(created by Fb) as its base and has many benefits with much more flexibilty than keras-tensorflow.
### Live detection made through Streamlit app using fastai
![image](https://user-images.githubusercontent.com/59556681/126079247-b3e3937a-6cf1-40f8-bfb8-9e2435f7476e.png)
![image](https://user-images.githubusercontent.com/59556681/126079252-7a209316-06f8-473c-a2ae-e4dfce708623.png)
![image](https://user-images.githubusercontent.com/59556681/126079259-8df83a67-1618-446f-b395-663e933aca8a.png)
### To run this model in your system follow the below system 
1. Go to some IDE like Pycharm or Visual studio and take the project files in the repository
2. Create a new environment
3. Run pip install -r requirements.txt
or
pip freeze > requirements.txt
4. You can run python app using liveemotion_detect.py file by just running python app.py or flask file in the project
5. To run streamlit app just run the below code
streamlit run streamlit_app.py 
### Deployment made in GCP and Streamlit sharing
- GCP link created from Kubernetes engine: http://35.202.208.253:8501/
- GCP Google cloud secure link: https://yamini-face-emo-recog-gcp-csyemhz4pq-uc.a.run.app
- Streamlit sharing link is not yet received will update it soon
- To see the work or application you can check with above links you can clear the cache and re-run the application if it does not work.
- To video to come up it may take few minutes if it doesn't come within 5 minutes(max) and any error pops up there may be some internal mistake.
- If it doesn't show you can please refer to the below demo link of the application
- Demo link:  https://youtu.be/JDKs6hsiCjo
-


