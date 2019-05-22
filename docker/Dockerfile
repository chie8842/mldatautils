FROM ubuntu:18.04
MAINTAINER Chie HAYASHIDA <chie-hayashida@cookpad.com>

ENV DEBIAN_FRONTEND=noninteractive
# Install Python3
RUN apt-get update && apt-get install -y \
  python3.6-minimal \
  python3-pip \
  language-pack-ja-base \
  language-pack-ja \
  postgresql \
  libpq-dev \
  git \
  curl \
  swig \
  cmake \
  sudo \
  mecab \
  libmecab-dev \
  mecab-ipadic \
  mecab-ipadic-utf8

## install mecab
#RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git /tmp/mecab-ipadic-neologd
#RUN /tmp/mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y
#RUN rm -rf /tmp/mecab-ipadic-neologd

# Update pip
RUN pip3 install --upgrade pip

## Install python modules.
#COPY ./requirements.txt /requirements.txt
#RUN pip3 install -r /requirements.txt

# Install python example modules.
COPY ./requirements_ex.txt /requirements_ex.txt
RUN pip3 install -r /requirements_ex.txt
RUN pip3 install mecab-python3

# settings
ARG user_name=ubuntu
ARG user_id=1000
ARG group_name=ubuntu
ARG group_id=1000

# create user
RUN groupadd -g ${group_id} ${group_name} \
  && \
    useradd -u ${user_id} -g ${group_id} -d /home/${user_name} --create-home --shell /bin/bash ${user_name} \
      && \
    echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
      && \
    chown -R ${user_name}:${group_name} /home/${user_name}

# user settings
USER ubuntu
WORKDIR /work
ENV HOME /home/ubuntu
ENV LANG="ja_JP.UTF-8" \
    LANGUAGE="ja_JP:ja" \
    LC_ALL="ja_JP.UTF-8"

# Set alias for python3.5
RUN echo "alias python=python3" >> $HOME/.bashrc && \
    echo "alias pip=pip3" >> $HOME/.bashrc

# Set working directory
WORKDIR /work

# Install python example modules.
COPY scripts/start-jupyter.sh /start-jupyter.sh

CMD ["/bin/bash", "/start-jupyter.sh"]
