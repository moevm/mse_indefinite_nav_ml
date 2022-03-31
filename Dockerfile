FROM python:3.8

COPY gym_custom ray
COPY tools/train.py ray
COPY tools/train.sh ray
COPY configs ray
COPY requirements.txt ray

WORKDIR ray

RUN apt-get update
RUN apt-get install -y xvfb
RUN apt-get install -y freeglut3-dev

RUN mkdir build
RUN cd build
RUN git clone https://github.com/duckietown/gym-duckietown.git
RUN python3 -m pip install -e gym-duckietown/.
RUN cd ../


RUN pip3 install  --user --upgrade pip
RUN pip3 install  --user -r requirements.txt

CMD ["bash", "train.sh"]
