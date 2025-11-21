FROM python:3.10-slim

ENV HOME="/home/hex"
ARG UID
RUN useradd -u $UID --create-home hex

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    default-jre \
    gcc \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# install requirements
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy scipy scikit-learn pandas


WORKDIR /home/hex
