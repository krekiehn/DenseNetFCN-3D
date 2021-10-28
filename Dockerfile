#FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04
#FROM python:3.8
FROM tensorflow/tensorflow:2.6.0-gpu

MAINTAINER "nicolai krekiehn <nicolai.krekiehn@rad.uni-kiel.de>"
#CMD nvidia-smi

WORKDIR /code


RUN apt-get update ##[edited]
#RUN apt-get install 'ffmpeg'\
#    'libsm6'\
#    'libxext6'  -y

#RUN apt-get install curl git unzip nano

RUN /usr/local/bin/python -m pip install --upgrade pip
ADD ./requirements.txt ./
RUN pip install -r requirements.txt
#RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install -y git
RUN cd .
RUN git clone https://github.com/krekiehn/DenseNetFCN-3D.git #code/DenseNet3D
RUN git clone https://www.github.com/keras-team/keras-contrib.git ./keras_contrib
#RUN cd keras_contrib
RUN python keras_contrib/convert_to_tf_keras.py
RUN python keras_contrib/setup.py install
# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]
CMD ["/bin/bash", "--login"]

