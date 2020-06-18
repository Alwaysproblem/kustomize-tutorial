FROM tensorflow/tensorflow

RUN pip install --no-cache-dir deepctr[cpu]

COPY model.py /
COPY config.json /

ENTRYPOINT ["python", "/model.py"]
