FROM python:3.11-slim

WORKDIR /app

COPY app ./app

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.main"]

