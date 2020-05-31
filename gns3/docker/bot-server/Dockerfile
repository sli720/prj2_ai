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
    pip install -r requirements.txt && \
    cd server && \
    ./ares.py initdb

WORKDIR /opt/Ares/server
COPY exec.py .
RUN chmod u+x exec.py
COPY commands.txt .
ENTRYPOINT ["./exec.py"]
