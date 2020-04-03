# Follow the openvpn instructions on cis to connect to the internal FH Technikum network
# Import your ssh key to the gns3 master (mcs20-paai password in the email from our lecturer)
ssh-copy-id mcs20-paai@mcs20-paai-n1.cs.technikum-wien.at
# Install ansible on your host
sudo apt install ansible
# Run the ansible playbook (mcs20-paai password in the email from our lecturer)
ansible-playbook -i hosts site.yml --ask-become-pass
# Connect to the gns3-server
./connect.sh
# Start gns3-gui
## All projects are stored on the gns3 server so you can work together
## Store docker images on docker hub (it is free for public projects)
