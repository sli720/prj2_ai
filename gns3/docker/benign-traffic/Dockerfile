FROM ubuntu:latest

ENV website="google.at"

RUN apt-get update && \
    apt-get install -y wget
    
ENTRYPOINT while true; do wget -O /dev/null ${website}; sleep 5; done
