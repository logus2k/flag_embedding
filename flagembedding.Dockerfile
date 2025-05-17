FROM flagembedding-server:1.0

USER root

COPY app.py /flagembedding

EXPOSE 8000

WORKDIR /flagembedding

CMD ["python", "app.py"]
