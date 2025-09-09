"""
Text extraction module for PDF and other document formats.
"""


import logging
from typing import Tuple


logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> Tuple[str, bool]:
    """
    Extract text from PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted_text, is_scanned)
    """
    try:
        # Try pdfplumber first
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            is_scanned = len(text.strip()) < 100  # Assume scanned if very little text
            return text.strip(), is_scanned
            
    except ImportError:
        logger.warning("pdfplumber not available, trying PyPDF2")
        try:
            import PyPDF2
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                is_scanned = len(text.strip()) < 100
                return text.strip(), is_scanned
                
        except ImportError:
            logger.error("No PDF processing libraries available")
            return "Error: No PDF processing libraries available", True
            
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return f"Error extracting text: {str(e)}", True


def detect_pdf_type(file_path: str) -> str:
    """
    Detect if PDF is text-based or scanned.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        'text' for text-based PDFs, 'scanned' for image-based PDFs, 'error' for processing errors
    """
    try:
        _, is_scanned = extract_text_from_pdf(file_path)
        return 'scanned' if is_scanned else 'text'
    except Exception as e:
        logger.error(f"Error detecting PDF type: {e}")
        return 'error'


def detect_language(text: str) -> str:
    """
    Detect language of text.
    
    Args:
        text: Input text
        
    Returns:
        Language code ('German' for German, 'English' for English)
    """
    if not text or len(text.strip()) < 10:
        return 'English'  # Default to English
    
    # Simple keyword-based language detection
    german_keywords = [
        'rechnung', 'bestellung', 'datum', 'betrag', 'gesamt', 'mwst', 
        'steuer', 'nummer', 'kunde', 'lieferant', 'zahlung', 'firma',
        'straße', 'plz', 'ort', 'deutschland', 'gmbh', 'ag'
    ]
    
    english_keywords = [
        'invoice', 'order', 'date', 'amount', 'total', 'tax', 'vat',
        'number', 'customer', 'supplier', 'payment', 'company',
        'street', 'address', 'city', 'country', 'inc', 'ltd', 'corp'
    ]
    
    text_lower = text.lower()
    
    german_score = sum(1 for keyword in german_keywords if keyword in text_lower)
    english_score = sum(1 for keyword in english_keywords if keyword in text_lower)
    
    if german_score > english_score:
        return 'German'
    elif english_score > german_score:
        return 'English'
    else:
        # Check for German-specific characters
        german_chars = ['ä', 'ö', 'ü', 'ß']
        if any(char in text_lower for char in german_chars):
            return 'German'
        
        return 'English'  # Default to English


# Export all functions for easy importing
__all__ = ['extract_text_from_pdf', 'detect_pdf_type', 'detect_language']

