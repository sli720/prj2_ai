FROM ubuntu:18.04
USER root
WORKDIR /opt

RUN apt-get update && \
    apt-get install -y \
      git \
      python-pip && \
    apt-get clean all

RUN git clone https://github.com/sweetsoftware/Ares.git && \
    cd Ares && \
    pip install -r requirements.txt

ARG SERVER=192.168.0.200
ARG PORT=8080
WORKDIR /opt/Ares/agent
RUN sed -i "s/localhost/$SERVER/g" ./config.py
RUN sed -i "s/8080/$PORT/g" ./config.py
ENTRYPOINT ["./agent.py"]
