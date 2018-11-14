FROM ubuntu:bionic

ENV DEBIAN_FRONTEND=noninteractive

# install menoh ruby
RUN mkdir -p /opt/src
RUN apt-get update && apt-get install -y \
  curl \
  wget \
  libprotobuf10 \
  git \
  gcc \
  g++ \
  cmake \
  cmake-data \
  libopencv-dev \
  protobuf-compiler \
  python \
  libprotobuf-dev \
  ruby-dev \
  ruby-rmagick \
  ruby-bundler

# install menoh
ADD https://github.com/pfnet-research/menoh/releases/download/v1.1.1/ubuntu1804_mkl-dnn_0.16-1_amd64.deb /opt/src
ADD https://github.com/pfnet-research/menoh/releases/download/v1.1.1/ubuntu1804_menoh_1.1.1-1_amd64.deb /opt/src
ADD https://github.com/pfnet-research/menoh/releases/download/v1.1.1/ubuntu1804_menoh-dev_1.1.1-1_amd64.deb /opt/src
RUN dpkg -i --force-depends /opt/src/*.deb
RUN gem install rake-compiler menoh

# install fluentd
RUN gem install fluentd

RUN rm -rf /var/lib/apt/lists/*
