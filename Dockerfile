FROM python:3.8

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /streamlit_app
ADD . /streamlit_app

RUN pip install --upgrade pip
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
'libsm6'\
'libxext6' -y

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN  apt-get install -y libgl1-mesa-dev
RUN xargs -a packages.txt apt-get install --yes

# Install dependencies
RUN pip install -r requirements.txt

# copying all files over
COPY . /streamlit_app

# Expose port
EXPOSE 8501

# cmd to launch app when container is run
CMD streamlit run streamlit_app.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'