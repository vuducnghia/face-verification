FROM python:3.6
WORKDIR /data


RUN apt-get update \
    && apt-get install -y \
    cmake


ENV distance=euclidean


RUN pip install --upgrade pip
RUN pip install face-recognition
RUN pip install \
    opencv-python==4.1.2.30 \
    scikit-image==0.16.2 \
    tensorflow==1.10.0 \
    keras==2.2.0 \
    gdown \
    h5py==2.10.0 \
    dlib

COPY demo_docker.py /data
COPY facenet_weights.h5 /data
COPY inception_resnet_v1.py /data
COPY images/IPS_2021_05_10_09_21_04_8840/img001.jpg /data/0.jpg
COPY images/IPS_2021_05_10_09_21_04_8840/img002.jpg /data/1.jpg
COPY shape_predictor_68_face_landmarks.dat /data
ENV thresh=0.2
ENTRYPOINT ["python", "demo_docker.py"]