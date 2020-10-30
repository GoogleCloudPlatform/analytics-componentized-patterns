FROM python:3.8-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . ./

ARG PORT
ENV PORT=$PORT

CMD exec gunicorn --bind :$PORT main:app  --workers=1 --threads 8 --timeout 1800