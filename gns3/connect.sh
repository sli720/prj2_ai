#!/bin/bash

ssh -L 1194:localhost:1194 mcs20-paai@mcs20-paai-n1.cs.technikum-wien.at -f sleep infinity
sudo openvpn client.ovpn