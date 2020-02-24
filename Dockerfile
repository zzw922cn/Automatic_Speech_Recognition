FROM		tensorflow/tensorflow:1.12.3-gpu-py3
MAINTAINER	sah0322@naver.com

RUN		apt-get -y update && apt-get -y install libsndfile1 libsndfile-dev python3-tk

RUN		pip install --upgrade pip
RUN		pip install six==1.11.0 \
				numpy==1.14.0 \
				matplotlib==2.0.2 \
				scikits.audiolab==0.11.0 \
				scipy==0.19.1 \
				scikit_learn==0.18.1 \
				tabulate==0.7.7 \
				theano==0.9.0 \
				xlwt==1.2.0 \
				librosa==0.5.1 \
				leven

WORKDIR		/opt/project
