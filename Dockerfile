FROM jupyter/scipy-notebook:0ce64578df46

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN jupyter labextension install @jupyterlab/toc

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc

WORKDIR /home/jovyan/work