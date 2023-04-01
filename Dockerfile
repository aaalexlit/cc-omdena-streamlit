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
RUN pip install --upgrade pip gdown && \
    pip install -r requirements.txt
RUN git clone https://github.com/dwadden/multivers.git ./multivers && \
    git clone https://github.com/yiweiluo/GWStance.git ./GWStance && \
    chmod +x check_gpu.py
RUN pip install virtualenv  && \
    virtualenv mult && \
    . mult/bin/activate &&  \
    pip install --upgrade pip && \
    pip install -r multivers/requirements.txt
RUN gdown https://drive.google.com/uc?id=12rVg_bpuDfZbdWRtEN2Jf6SNyMEnax76 && \
    tar -xvzf final_model.tar.gz
RUN python multivers/script/get_checkpoint.py longformer_large_science && \
    python multivers/script/get_checkpoint.py fever_sci && \
    mv checkpoints multivers/
RUN cp -r GWStance/3_stance_detection/2_Stance_model/for_transformers/* /opt/bitnami/python/lib/python3.8/site-packages/transformers/
COPY app/streamlit_app.py ./streamlit/
COPY start.sh ./
RUN chmod +x start.sh
ENV EVIDENCE_API_IP=
HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health
ENTRYPOINT ["./start.sh"]