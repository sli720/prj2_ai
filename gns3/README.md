# Server credentials
hostname: <hostname>
username: <username>
password: <password>

# Setup GNS3 server
* Connect to FH via VPN
* Run ansible-playbook -i hosts setup.yml --ask-become-pass
* This will install GNS3 server, setups OpenVPN and copies the GNS3 projects to the server
* The ansible playbook will download the OpenVPN client configuration to your pc

# Connect to GNS3
* Connect to FH via VPN
* Run ./connect to connect to the FH server directly via another VPN (required by GNS3)
* Install and start GNS3 GUI on your pc
* Goto Edit->Preferences->Server->Main server
* Disable local server
* Set Host to 172.16.253.1
* Set Port to 3080 TCP
* Wait till the server appears in the Servers Summary on the right side
* Goto File->Open project->Projects library
* Select the project you want to modify and press OK to open it
* Now you can modify and simulate the project

# Save project to ansible 
* Projects are automatically saved on the server
* To submit the project to ansible connect to the server via ssh <username>@<hostname>
* Goto /opt/gns3/projects
* Archive the project with tar -cvzf <project>
* Download the archive to your pc via scp <username>@<hosntame>:/opt/gns3/projects/<project>.tar.gz <destination>
* Extract the folder and submit the files to gns3/projects/<project>

# Run simulation via ansible
* Run ansible-playbook -i hosts run.yml -e duration=<minutes> -e project=<project> -e pcap=<true|false> --ask-become-pass

