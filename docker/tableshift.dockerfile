FROM python:3.8-bullseye

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install git
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

RUN mkdir /tableshift
COPY . /tableshift
WORKDIR /tableshift
RUN python -m pip install --no-deps .

# Add tableshift to pythonpath; necessary to ensure
# tableshift module imports work inside docker.
ENV PYTHONPATH "${PYTHONPATH}:/tableshift"

ENTRYPOINT [ "python", "examples/run_expt.py"]