# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    bc \
    wget \
    bzip2 \
    openssh-server \
    iproute2 \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-distutils \
    python3-pip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

# Install Python dependencies
RUN pip install \
    torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install \
    dgl==0.9 \
    pandas \
    scikit-learn \
    matplotlib \
    ogb \
    "numpy<2" \
    pymetis


# Set the environment variable for DGL
ENV DGLBACKEND=pytorch

# Configure SSH server
RUN mkdir /var/run/sshd && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'root:root' | chpasswd

# Expose SSH port
EXPOSE 22

# Default command to start SSH server
CMD ["/usr/sbin/sshd", "-D"]
