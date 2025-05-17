FROM python:3.12.10-slim-bookworm

USER root

RUN apt update && apt upgrade && apt autoremove
RUN pip install --upgrade pip
RUN pip install FlagEmbedding fastapi uvicorn

WORKDIR /flagembedding
