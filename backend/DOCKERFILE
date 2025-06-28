FROM tensorflow/tensorflow

WORKDIR .

COPY . .

RUN pip install -r requirements.txt
RUN pip install --upgrade --quiet duckduckgo-search

# Populate the vector DB if need:
CMD ["sh", "-c", "uvicorn main:app --reload"]