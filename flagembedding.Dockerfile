FROM python:3.12.10-slim-bookworm

USER root

RUN apt update && apt upgrade && apt autoremove
RUN pip install --upgrade pip
RUN pip install FlagEmbedding fastapi uvicorn

WORKDIR /flagembedding

COPY app.py /flagembedding

EXPOSE 8000

WORKDIR /flagembedding

CMD ["python", "app.py"]
