FROM ubuntu:latest

RUN apt-get update && \ 
    apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd && \
    useradd technikum -m -d /home/technikum && \
    echo technikum:technikum | chpasswd


EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

