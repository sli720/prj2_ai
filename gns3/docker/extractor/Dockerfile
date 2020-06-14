#First stage: build CICFlowMeter using gradle
FROM gradle:6.3.0-jdk11 as build

#Copy CICFlowMeter from build context (i.e., the current working directory when running "docker build")
COPY CICFlowMeter /CICFlowMeter
#Build the CICFlowMeter distribution
WORKDIR /CICFlowMeter
RUN gradle assembleDist
#Extract the tar file that was just built by gradle
RUN tar -xf build/distributions/CICFlowMeter-*.tar -C /CICFlowMeter

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
COPY --from=build /CICFlowMeter/CICFlowMeter-* CICFlowMeter

ENTRYPOINT /usr/share/java/bin/java \
	#The java libraries are placed in lib, the * tells java to look into every jar file in that directory:
	-classpath CICFlowMeter/lib/\* \
	#Native libraries are placed in lib/native:
	-Djava.library.path=CICFlowMeter/lib/native \
	#The main class for the command-line interface of CICFlowMeter:
	cic.cs.unb.ca.ifm.Cmd \
	#The location of the pcap directory:
	/mnt/packet-data \
	#The directory where the resulting csv file is stored
	/mnt/packet-data \
	#The attack labels (separated by ':') are read from environment variable ATTACKS
	$ATTACKS \
	#The ignored ethernet types (separated by ':') are read from environment variable IGNORE
	"$IGNORE" \
	#The extracted features (separated by ',') are read from environment variable FEATURES
	"$FEATURES"
