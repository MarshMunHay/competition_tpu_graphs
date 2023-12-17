FROM python:3.9.11-buster as python
WORKDIR /project

RUN apt-get update && apt-get -y install cmake protobuf-compiler

COPY ./requirements.txt /project/
RUN pip install -r requirements.txt
