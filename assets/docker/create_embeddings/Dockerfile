FROM python:3.10-slim-bullseye

WORKDIR /embed

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
