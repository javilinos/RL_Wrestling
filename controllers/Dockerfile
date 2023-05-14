FROM cyberbotics/webots.cloud:R2023a-ubuntu20.04-numpy

# Additional dependencies for the utils folder
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt install wget 

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda

RUN pip3 install --no-cache-dir torch torchvision torchaudio

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    scipy \
    ahrs \
    torch \
    torchvision \
    torchaudio \
    stable-baselines3 \
    sb3-contrib \
    gymnasium \
    pytransform3d 

# Environment variables needed for Webots
# https://cyberbotics.com/doc/guide/running-extern-robot-controllers#remote-extern-controllers
ENV PYTHONPATH=${WEBOTS_HOME}/lib/controller/python
ENV PYTHONIOENCODING=UTF-8
ENV LD_LIBRARY_PATH=${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}
ARG WEBOTS_CONTROLLER_URL
ENV WEBOTS_CONTROLLER_URL=${WEBOTS_CONTROLLER_URL}
ENV NVIDIA_DRIVER_CAPABILITIES=graphics

# Copies all the files of the controllers folder into the docker container
RUN mkdir -p /usr/local/webots-project/controllers
COPY . /usr/local/webots-project/controllers

# Entrypoint command to launch default Python controller script
# (In shell form to allow variable expansion)
WORKDIR /usr/local/webots-project/controllers/participant
ENTRYPOINT python3 -u participant.py
