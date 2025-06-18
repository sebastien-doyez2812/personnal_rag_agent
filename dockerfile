FROM tensorflow/tensorflow

WORKDIR .

COPY . .

RUN pip install -r requirements.txt


# Populate the vector DB if need:
CMD ["sh", "-c", "python populate_vectordb.py && python app.py"]