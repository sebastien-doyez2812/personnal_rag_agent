FROM python:3.12.11-alpine3.21

WORKDIR .

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]