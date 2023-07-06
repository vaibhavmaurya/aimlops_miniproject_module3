#FROM alpine:latest AS builder
FROM python:3.9

COPY dist/bikeshare_model-1.0.0-py3-none-any.whl /etc/bikeshare_model-1.0.0-py3-none-any.whl 
COPY bikeshare_model_api /etc/bikeshare_model_api

RUN pip3 install -r /etc/bikeshare_model_api/requirements.txt
RUN pip3 install /etc/bikeshare_model-1.0.0-py3-none-any.whl

EXPOSE 8000

WORKDIR /etc/bikeshare_model_api
CMD ["uvicorn" , "api:app",  "--host", "0.0.0.0", "--reload"]

