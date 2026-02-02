import os
import sys
from pathlib import Path
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "pdf_documents"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():  
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def process_pdfs(pdf_dir):
    """Process all PDFs in a directory and return chunks with metadata."""
    pdf_dir = Path(pdf_dir)
    all_chunks = []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        text = extract_text_from_pdf(pdf_file)
        
        if text:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "filename": pdf_file.name,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
    
    return all_chunks

def create_collection(client, embedding_dim):
    """Create or recreate the Qdrant collection."""
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )
    print(f"Created collection: {COLLECTION_NAME}")

def index_documents(chunks, client, model):
    """Index document chunks into Qdrant."""
    print(f"Indexing {len(chunks)} chunks...")
    
    batch_size = 100
    points = []
    
    for idx, chunk_data in enumerate(tqdm(chunks, desc="Creating embeddings")):
        # Generate embedding
        embedding = model.encode(chunk_data["text"]).tolist()
        
        # Create point
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "text": chunk_data["text"],
                "filename": chunk_data["filename"],
                "chunk_id": chunk_data["chunk_id"],
                "total_chunks": chunk_data["total_chunks"]
            }
        )
        points.append(point)
        
        # Upload in batches
        if len(points) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    print(f"Successfully indexed {len(chunks)} chunks")

def main():
    # Wait for Qdrant to be ready
    print("Waiting for Qdrant to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()
            print("Connected to Qdrant!")
            break
        except Exception as e:
            if i < max_retries - 1:
                print(f"Attempt {i+1}/{max_retries}: Qdrant not ready, waiting...")
                time.sleep(2)
            else:
                print(f"Failed to connect to Qdrant after {max_retries} attempts")
                sys.exit(1)
    
    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")
    
    # Process PDFs
    pdf_dir = "/data"
    chunks = process_pdfs(pdf_dir)
    
    if not chunks:
        print("No chunks to index!")
        return
    
    # Create collection
    create_collection(client, embedding_dim)
    
    # Index documents
    index_documents(chunks, client, model)
    
    print("Indexing complete!")

if __name__ == "__main__":
    main()
