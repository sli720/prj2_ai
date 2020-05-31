FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y goldeneye

ENTRYPOINT ["goldeneye", "http://192.168.0.100"]
