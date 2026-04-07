FROM python:3.11-slim

WORKDIR /home/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nano unzip curl \
    && rm -rf /var/lib/apt/lists/*

COPY . . 

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

