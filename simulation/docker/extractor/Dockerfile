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


#Second stage: build a custom JRE to reduce size of final image
FROM azul/zulu-openjdk:11 as custom_jre
RUN jlink \
	# Add only the modules required by CICFlowMeter:
	--add-modules java.base,java.xml,java.sql,java.desktop \
	# Use ZIP compression:
	--compress 2 \
	# Don't include debug information, header files and man pages:
	--strip-debug \
	--no-header-files \
	--no-man-pages \
	# The output directory of the resulting JRE:
	--output /minimal_jre


#Final stage: run previously built CICFlowMeter
#Note: jnetpcap doesn't work with alpine linux because of missing native dependencies, so we use Ubuntu
FROM ubuntu:latest
#install libpcap, because it is a runtime dependency of CICFlowMeter
RUN apt-get update && apt-get install -y libpcap-dev
#Copy the custom JRE from the second stage
COPY --from=custom_jre /minimal_jre /usr/share/java
#Copy the CICFlowMeter distribution from the first stage
COPY --from=build /CICFlowMeter-master/CICFlowMeter-* CICFlowMeter

ENTRYPOINT /usr/share/java/bin/java \
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
