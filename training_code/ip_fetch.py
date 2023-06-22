import os
cmd = 'scontrol show hostnames'
os.system(cmd)
import time

with open('vgg.txt') as f:
    lines = f.readlines()
new_lines = [''.join([x.strip(), ".iitk.ac.in"]) for x in lines]
with open('abc.txt', 'w') as f:
    for j in new_lines:
        f.writelines(os.popen('cat /etc/hosts | grep -i {}'.format(j)).read())
os.system("cut -d' ' -f1 abc.txt >> ip_config.txt")
