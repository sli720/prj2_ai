FROM ubuntu:latest

ENV targetip=192.168.1.100
ENV command="echo test; echo test2; echo test3; mkdir test; rmdir test;"

RUN apt-get update && \ 
    apt-get install -y sshpass
    
ENTRYPOINT while true; do sshpass -p "technikum" ssh -o StrictHostKeyChecking=no "technikum@${targetip}" ${command}; sleep 5; done
