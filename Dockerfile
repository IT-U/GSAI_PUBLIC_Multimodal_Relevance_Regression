# Small Python 3.12 image
FROM python:3.12-slim AS runtime

# Working directory
WORKDIR /workspace

# Install required tools for microsoft font installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        cabextract \
        xz-utils \
        fontconfig && \
    rm -rf /var/lib/apt/lists/*

# Download and extract Microsoft Core Fonts manually
RUN mkdir -p /usr/share/fonts/truetype/msttcorefonts && \
    cd /usr/share/fonts/truetype/msttcorefonts && \
    wget -q https://downloads.sourceforge.net/corefonts/arial32.exe && \
    wget -q https://downloads.sourceforge.net/corefonts/arialb32.exe && \
    cabextract arial32.exe && \
    cabextract arialb32.exe && \
    rm arial32.exe arialb32.exe && \
    fc-cache -f -v

# Install Python dependencies from requirements.txt file
RUN python -m pip install --upgrade pip
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt