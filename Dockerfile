FROM bitnami/pytorch:1.13.1
USER root
ENV PIP_NO_CACHE_DIR=1 \
 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update &&  \
    apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git  \
    bash  \
    wget
WORKDIR /app
COPY requirements.txt ./
COPY check_gpu.py ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
RUN git clone https://github.com/dwadden/multivers.git ./multivers && \
    python check_gpu.py
RUN pip install virtualenv  && \
    virtualenv mult && \
    . mult/bin/activate &&  \
    pip install --upgrade pip && \
    pip install -r multivers/requirements.txt && \
    python multivers/script/get_checkpoint.py longformer_large_science && \
    python multivers/script/get_checkpoint.py fever_sci && \
    mv checkpoints multivers/
COPY app/streamlit_app.py ./
ENV EVIDENCE_API_IP=
HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=80", "--server.address=0.0.0.0"]