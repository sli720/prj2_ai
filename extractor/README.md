# Build the extractor docker image:
docker build extractor -t extractor

# Run the extractor docker image
## (packet-data directory has to contain a file called packets.pcap, and packets.pcap_Flow.csv will be written there)
docker run --mount type=bind,source=/path-to-packet-data-directory,destination=/mnt/packet-data extractor