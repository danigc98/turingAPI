FROM python:3.11.6-slim-bookworm
LABEL authors="danigc"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 80
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
