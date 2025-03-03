FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

# Install dependencies
# not sure why we still need nvidia-cuda-toolkit, but without it, we get error when running pytorch inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    ca-certificates \
    python3.10 \
    python3.10-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 - \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && update-alternatives --install /usr/bin/python3 python /usr/bin/python3.10 1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "Python installed at: $(which python3)"

RUN python -m pip --version

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workdir