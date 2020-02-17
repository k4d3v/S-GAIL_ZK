FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu16.04 
FROM continuumio/anaconda3

USER root

RUN conda install -y tensorflow-gpu

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute 


ENV DEBIAN_FRONTEND=noninteractive


# (1-1) Install prerequisites
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        wget \
        vim \
        kmod \
        libc6:i386 \
        libelf-dev && \
    rm -rf /var/lib/apt/lists/*

# (1-2) Install xorg server and xinit BEFORE INSTALLING NVIDIA DRIVER.
# After this installation, command Xorg and xinit can be used in the container
RUN apt-get update && apt-get install -y \
        xinit && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        mesa-utils \
        x11vnc \
        x11-apps && \
    rm -rf /var/lib/apt/lists/*

# (2-2) Optional vulkan support
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 vulkan-utils && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
        qt5-default qtcreator && \
    rm -rf /var/lib/apt/lists/*


RUN pip install wrapt --upgrade --ignore-installed


#ARG USER_ID=1001
#ARG GROUP_ID=1001
#ENV USER_NAME=developer
#RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
#    useradd -d /home/${USER_NAME} -m -s /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}
#WORKDIR /home/${USER_NAME}
#USER ${USER_NAME}
#ENV HOME /home/${USER_NAME}


RUN git clone https://github.com/benelot/pybullet-gym.git && \
cd pybullet-gym && \
pip install -e . 

RUN pip install h5py
RUN pip install gym
RUN pip install torch

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0","--NotebookApp.token=''"]

