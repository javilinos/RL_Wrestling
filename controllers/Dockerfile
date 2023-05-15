FROM cyberbotics/webots.cloud:R2023a-ubuntu20.04-numpy

# Additional dependencies for the utils folder
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch torchvision torchaudio

# Enable OpenGL capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=graphics

RUN apt update && apt-get install python3-pip -y

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir torch torchvision torchaudio
# Additional dependencies for the utils folder
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    opencv-python \
    scipy \
    ahrs \
    stable-baselines3==2.0.0a5 \
    sb3-contrib==2.0.0a4 \
    gymnasium \
    pytransform3d 

# Environment variables needed for Webots
# https://cyberbotics.com/doc/guide/running-extern-robot-controllers#remote-extern-controllers
ENV PYTHONPATH=${WEBOTS_HOME}/lib/controller/python
ENV PYTHONIOENCODING=UTF-8
ENV LD_LIBRARY_PATH=${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}
ARG WEBOTS_CONTROLLER_URL
ENV WEBOTS_CONTROLLER_URL=${WEBOTS_CONTROLLER_URL}

# Copies all the files of the controllers folder into the docker container
RUN mkdir -p /usr/local/webots-project/controllers
COPY . /usr/local/webots-project/controllers

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Entrypoint command to launch default Python controller script
# (In shell form to allow variable expansion)
WORKDIR /usr/local/webots-project/controllers/participant
ENTRYPOINT python3 -u participant.py