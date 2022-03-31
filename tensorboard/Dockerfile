FROM python:3.8

RUN pip3 install  --user --upgrade pip
RUN pip3 install --user tensorboard

COPY run_tensorboard.sh .
CMD ["bash", "run_tensorboard.sh"]