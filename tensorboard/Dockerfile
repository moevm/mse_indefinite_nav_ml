FROM python:3.8

RUN pip3 install  --user --upgrade pip
RUN pip3 install  --user tensorflow
RUN pip3 install --user tensorboard

COPY test test
WORKDIR test
CMD ["bash", "run_tensorboard.sh"]