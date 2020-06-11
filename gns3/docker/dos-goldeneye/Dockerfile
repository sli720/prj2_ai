FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y \
        goldeneye \
        expect

#There is a bug in goldeneye which needs to be fixed by adding a cast to int() in the following line of code:
RUN sed -i -E 's/nrEncodings = random\.randint\(1,len\(acceptEncoding\)\/2\)/nrEncodings = random.randint(1, int(len(acceptEncoding)\/2))/g' /usr/bin/goldeneye

COPY dos-goldeneye.expect ./

ENTRYPOINT "expect" "dos-goldeneye.expect"
