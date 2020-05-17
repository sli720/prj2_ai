FROM alpine

RUN apk add --update --no-cache tcpdump && rm -rf /var/cache/apk/*

CMD /usr/sbin/tcpdump -C 1000 -v -i eth0 -e $TCPD_EXTFILTER -w $TCPD_FILENAME
