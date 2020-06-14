FROM alpine

RUN apk add --update --no-cache tcpdump && rm -rf /var/cache/apk/*

ENV TCPD_FILENAME="/tmp/default.pcap"
ENV TCPD_EXTFILTER=""
CMD /usr/sbin/tcpdump -v -i eth0 -e -w $TCPD_FILENAME $TCPD_EXTFILTER
