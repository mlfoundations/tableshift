This repo contains a dockerfile used to build the official TableShift docker container.

To run existing baselines, we recommend that you pull the TableShift container from Docker Hub via:

```
docker pull
```

However, if you would like to build from this image and construct your own models/baselines, you can do so folowing the steps below.

## Building the image locally

Modify the `tableshift.dockerfile` to build your environment as desired. Then, from the project root (one level u from the `docker` directory!) build the image via:

```
docker build -t tableshift -f docker/tableshift.dockerfile .
```

## Running the container

We describe two cases below: one where you would simply like to run a sample script to verify that

**To test your installation**: If you'd simply like to run the docker container to verify that your setup is working, you can do that via

```
docker run -v $(pwd)/tmp:/tableshift/tmp tableshift --model lightgbm
```

You should see lots of logging output, and a message containing computed test accuracy when the execution completes. If this works, your setup is good to go! You can change `model` to any of the other model names described in the main README.

**To run experiments interactively:** To run the docker container interactively after either pulling it (via `docker pull`) or building it (via `docker build`), simply do:

```
docker run -it --entrypoint=/bin/bash -v $(pwd)/tmp:/tableshift/tmp tableshift
```

