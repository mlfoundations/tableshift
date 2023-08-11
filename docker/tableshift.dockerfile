FROM python:3.8-bullseye

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install git
COPY requirements.txt requirements.txt
RUN python -m pip install -r  requirements.txt