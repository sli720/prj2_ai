# Build the extractor docker image:
#### assuming that the current working directory is the extractor directory, therefore using '.':
docker build . -t extractor

# Run the extractor docker image
#### packet-data directory has to contain files named attacktype*.pcap (all other files are treated as benign traffic), and features_Flow.csv will be written there, the attack names are put in environment variable ATTACKS separated by ':'
docker run --mount type=bind,source=/path-to-packet-data-directory,destination=/mnt/packet-data --env ATTACKS=ssh-bruteforce:loic extractor