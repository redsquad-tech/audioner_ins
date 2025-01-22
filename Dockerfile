FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
RUN rm /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list && \
    apt-get update && \
    apt-get install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install python3.11 python3.11-distutils ffmpeg && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip install nvidia-cublas-cu11 nvidia-cudnn-cu12


WORKDIR /app
#COPY src src
#COPY .streamlit .streamlit
#COPY streamlit.py ./
ENTRYPOINT streamlit run streamlit.py
