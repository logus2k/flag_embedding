FROM flagembedding-server:1.0

USER root

COPY flagembedding.py /flagembedding

EXPOSE 8000

WORKDIR /flagembedding

CMD ["python", "-u", "flagembedding.py"]
