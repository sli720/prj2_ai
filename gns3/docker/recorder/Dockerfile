FROM alpine

RUN apk add --update --no-cache tcpdump && rm -rf /var/cache/apk/*

CMD /bin/sh -c "/usr/sbin/tcpdump -v -i eth0 -e -w $FILENAME '$FILTER'"
