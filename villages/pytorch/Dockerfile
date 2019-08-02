FROM pytorch/pytorch
MAINTAINER ductricse@gmail.com
ENV REFRESHED_AT 2019-03-26

RUN apt update
RUN apt install -y vim
RUN apt install -y wget

RUN pip install nltk
RUN pip install joblib==0.12.4
RUN pip install gensim==3.5.0
RUN pip install requests==2.18.4
RUN pip install pygtrie==2.3
RUN pip install tldextract==2.2.0
RUN python -m nltk.downloader 'punkt'

RUN pip install jupyter_contrib_nbextensions
RUN pip install jupyter_nbextensions_configurator
RUN jupyter contrib nbextension install
RUN jupyter nbextensions_configurator enable --user

RUN pip install --upgrade matplotlib==3.0.2
RUN pip install dask==1.1.4
RUN pip install dask[dataframe]==1.1.4

RUN pip install sklearn
RUN pip install Cython==0.27.3
RUN pip install seqlearn==0.2

RUN git clone https://github.com/OpenNMT/OpenNMT-py.git && cd OpenNMT-py && pip install -r requirements.txt && python setup.py install

RUN pip install https://github.com/kpu/kenlm/archive/master.zip

RUN pip install torchsummary
RUN pip install pytorch-nlp==0.4.1
RUN pip install tensorflow==1.13.1
RUN pip install tensorboard==1.13.1
RUN pip install tensorboardX==1.7
RUN pip install sentencepiece==0.1.82

# Install naruto_skills
ADD skills/ /tmp/skills/
RUN pip install /tmp/skills/

VOLUME /source
VOLUME /dataset

WORKDIR /source

# Jupyter Config
RUN jupyter nbextension enable collapsible_headings/main

