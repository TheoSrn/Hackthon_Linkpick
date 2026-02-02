FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy indexing script
COPY index_documents.py .

CMD ["python", "index_documents.py"]
