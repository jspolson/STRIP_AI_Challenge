From nvcr.io/nvidia/pytorch:19.06-py3

ARG USER_ID
ARG GROUP_ID

# install opensilde
RUN apt-get update --fix-missing && \
  apt-get install -y python3-openslide


# install ssh
RUN apt install -y --no-install-recommends openssh-server

#install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install torch==1.7.0 torchvision==0.11.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN conda install captum -c pytorch
RUN pip install spams seaborn scikit-image scikit-learn
RUN conda install -c conda-forge tqdm
RUN conda install -c conda-forge ipywidgets
RUN conda install opencv pixman=0.36.0
RUN conda install -c pytorch -c fastai fastai
RUN pip install h5py lmdb openslide-python tensorboardX Pillow tensorboard
RUN pip install --upgrade jupyter_client
RUN pip install tifffile

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} docker-user &&\
    useradd -l -u ${USER_ID} -g docker-user docker-user &&\
    install -d -m 0755 -o docker-user -g docker-user /home/docker-user \
;fi

RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.port = 4111" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = 'uclacdx'" >> ~/.jupyter/jupyter_notebook_config.py

USER docker-user