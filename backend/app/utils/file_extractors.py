"""Utilitaires d'extraction de texte de fichiers."""
import io
from fastapi import HTTPException
import PyPDF2
import docx


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extraire le texte d'un fichier PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")


def extract_text_from_docx(file_content: bytes) -> str:
    """Extraire le texte d'un fichier DOCX."""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")
