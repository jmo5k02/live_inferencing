FROM ultralytics/ultralytics:latest

# Install additional packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    curl \
    unzip
    
RUN pip install --upgrade pip

RUN pip install jupyterlab fastapi opencv-python uvicorn

WORKDIR /home/workspace

COPY ./main.py .

COPY ./data ./data