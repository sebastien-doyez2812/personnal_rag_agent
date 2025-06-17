FROM python:3.12.11-alpine3.21

WORKDIR .

COPY . .

RUN pip install -r requirements.txt


# Populate the vector DB if need:
CMD ["sh", "-c", "python populate_vectordb.py && python app.py"]