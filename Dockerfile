FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    wget \
    git \
    ffmpeg \
    libaom-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3 -m pip install --upgrade pip

# Install dependencies
RUN pip3 install torch torchvision torchaudio
RUN pip3 install https://github.com/ytdl-org/youtube-dl/archive/refs/heads/master.zip
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "web.py", "--server.port=8501", "--server.address=0.0.0.0"]
