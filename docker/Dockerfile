FROM ubuntu:18.04

# Using aliyun ubuntu mirror
#RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
#RUN apt update

RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu1804_install_python.sh /install/ubuntu1804_install_python.sh
RUN bash /install/ubuntu1804_install_python.sh

# Globally disable pip cache
RUN pip config set global.cache-dir false

COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

# Python: basic dependencies
#RUN sudo apt-get install -y software-properties-common
#RUN sudo add-apt-repository -y ppa:jblgf0/python && sudo apt-get update && sudo rm /usr/bin/python3 && sudo apt-get install -y python3.7
#RUN ln -s /usr/bin/python3.7 /usr/bin/python3
#RUN ln -s /usr/bin/python3.7 /usr/bin/python
#RUN sudo apt-get install -y python3-dev python3-setuptools gcc \
#         libtinfo-dev zlib1g-dev build-essential cmake libboost-all-dev

#RUN sudo apt-get install -y python3-pip
#RUN sudo pip3 install --upgrade pip

# LLVM
RUN mkdir /llvm && wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
RUN tar xvf clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
RUN rm clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && mv clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/ /llvm/ 

# Other tools
RUN apt-get install -y vim 
# PythonCppDubugger
RUN apt-get -y install gdb
RUN apt-get -y install ssh
RUN apt-get -y install gcc


RUN mkdir /workspace

# Environment variables
ENV TVM_HOME=/workspace/gpt-frontend/tvm
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV LLVM_DIR=/llvm/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04

# Using douban pipy mirror
RUN pip3 install -i https://pypi.douban.com/simple/ -U pip
RUN pip3 config set global.index-url https://pypi.douban.com/simple/

# Vision transformer pytorch
RUN pip3 install vit-pytorch
RUN pip3 install termcolor
RUN pip3 install timm
RUN pip3 install yacs

# Deep learning framework dependencies
#RUN pip3 install torchvision==0.4.2
#RUN pip3 install onnx==1.6.0
RUN pip3 install transformers
RUN pip3 install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install psutil
#RUN pip3 install torch
#RUN pip3 install tensorflow==1.15
#RUN pip3 install keras=2.2.4
#RUN pip3 install h5py=2.10.0
# RUN pip3 install pytorch_pretrained_vit

RUN pip3 install ipdb

#update cmake
RUN wget https://cmake.org/files/v3.21/cmake-3.21.2-linux-x86_64.tar.gz \
    && tar -zxvf cmake-3.21.2-linux-x86_64.tar.gz \
    && mv cmake-3.21.2-linux-x86_64 cmake-3.21.2 \
    && ln -sf /cmake-3.21.2/bin/* /usr/bin
