FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

RUN rm -f /etc/apt/sources.list.d/*.list

# ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends\
    vim \
    curl \
    apt-utils \
    ssh \
    tree \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Seoul
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config

# Re-run ssh when the container restarts.
RUN echo "sudo service ssh start > /dev/null" >> $HOME/.bashrc

# Create a workspace directory
RUN mkdir $HOME/workspace
WORKDIR $HOME/workspace

ARG PYTHON_VERSION 3.10.6
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH "$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV eval "$(pyenv init -)"
RUN cd $HOME && /bin/bash -c "source .bashrc" && \
    /bin/bash -c "pyenv install -v $PYTHON_VERSION" && \
    /bin/bash -c "pyenv global $PYTHON_VERSION"

ENV PATH "$HOME/.local/bin:$PATH"
ENV PYTHON_KEYRING_BACKEND keyring.backends.null.Keyring
RUN curl -sSL https://install.python-poetry.org | python - && \
    poetry config virtualenvs.in-project true && \ 
    poetry config virtualenvs.path "./.venv"

WORKDIR /workspace
COPY ./requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt
COPY . /workspace