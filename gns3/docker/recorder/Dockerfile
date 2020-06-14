FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y tcpdump && \
    apt-get clean all

ENTRYPOINT /usr/sbin/tcpdump -v -i eth0 -e -w $TCPD_FILENAME $TCPD_EXTFILTER
