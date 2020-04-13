#First stage: build CICFlowMeter using gradle
FROM gradle:6.3.0-jdk11 as build

#Download and unzip CICFlowMeter from github
RUN wget https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter/archive/master.zip
RUN unzip master.zip -d /
WORKDIR /CICFlowMeter-master
#We have to replace the version of jnetpcap in the build file, because 1.4.1 doesn't exist in the clojars.org repository anymore - instead 1.4.r1425-1g is the newest version
RUN sed -i "s/'org.jnetpcap', name: 'jnetpcap', version:'1.4.1'/'jnetpcap', name: 'jnetpcap', version:'1.4.r1425-1g'/" build.gradle
#Build the CICFlowMeter distribution
RUN gradle assembleDist
#Extract the tar file that was just built by gradle
RUN tar -xf build/distributions/CICFlowMeter-*.tar -C /CICFlowMeter-master

#Second stage: run previously built CICFlowMeter
FROM azul/zulu-openjdk:11
#install libpcap, because it is a runtime dependency of CICFlowMeter
RUN apt-get update && apt-get install -y libpcap-dev
#Copy the CICFlowMeter distribution from the previous stage
COPY --from=build /CICFlowMeter-master/CICFlowMeter-* CICFlowMeter

ENTRYPOINT java \
#The java libraries are placed in lib, the * tells java to look into every jar file in that directory:
-classpath CICFlowMeter/lib/\* \
#Native libraries are placed in lib/native:
-Djava.library.path=CICFlowMeter/lib/native \
#The main class for the command-line interface of CICFlowMeter:
cic.cs.unb.ca.ifm.Cmd \
#The location of the pcap file:
/mnt/packet-data/packets.pcap \
#The directory where the resulting csv file is stored
/mnt/packet-data