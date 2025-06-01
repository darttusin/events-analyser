FROM python:3.10-slim

WORKDIR /mlserver

COPY requirements.txt . 
RUN pip install -r requirements.txt

CMD ["mlserver", "start", "."]
