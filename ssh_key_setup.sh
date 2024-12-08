
# ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" run this command to generate key on container1
mkdir -p /root/.ssh
echo "<paste-the-public-key-here>" >> /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
