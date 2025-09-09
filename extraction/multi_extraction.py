import tempfile
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging



logger = logging.getLogger(__name__)



class MultiFormatExtractor:
    """Extract text and data from multiple file formats."""
   
    def __init__(self):
        self.supported_formats = {
            'pdf': self._extract_from_pdf,
            'png': self._extract_from_image,
            'jpg': self._extract_from_image,
            'jpeg': self._extract_from_image,
            'tiff': self._extract_from_image,
            'bmp': self._extract_from_image,
            'docx': self._extract_from_docx,
            'doc': self._extract_from_doc,
            'txt': self._extract_from_text,
            'json': self._extract_from_json,
            'xml': self._extract_from_xml,
            'csv': self._extract_from_csv,
            'xlsx': self._extract_from_excel,
            'xls': self._extract_from_excel
        }
   
    def extract_from_file(self, file_content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and structured data from uploaded file.
       
        Args:
            file_content: Raw file bytes
            filename: Original filename with extension
           
        Returns:
            Tuple of (extracted_text, metadata)
        """
        file_ext = Path(filename).suffix.lower().lstrip('.')
       
        if file_ext not in self.supported_formats:
            return f"Unsupported file format: {file_ext}", {
                'error': f'File format .{file_ext} is not supported',
                'supported_formats': list(self.supported_formats.keys())
            }
       
        try:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
           
            # Extract based on format
            extractor_func = self.supported_formats[file_ext]
            text, metadata = extractor_func(tmp_path, file_content)
           
            # Add common metadata
            metadata.update({
                'original_filename': filename,
                'file_type': file_ext,
                'file_size': len(file_content)
            })
           
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
           
            return text, metadata
           
        except Exception as e:
            logger.error(f"Error extracting from {filename}: {e}")
            return f"Error processing {filename}: {str(e)}", {
                'error': str(e),
                'file_type': file_ext
            }
   
    def _extract_from_pdf(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract text from PDF files."""
        try:
            # Try pdfplumber first
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
               
                return text.strip(), {
                    'pages': len(pdf.pages),
                    'method': 'pdfplumber'
                }
        except ImportError:
            pass
       
        try:
            # Fallback to PyPDF2
            import PyPDF2
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
               
                return text.strip(), {
                    'pages': len(reader.pages),
                    'method': 'PyPDF2'
                }
        except ImportError:
            return "PDF processing libraries not available", {'error': 'No PDF library found'}
   
    def _extract_from_image(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract text from images using OCR."""
        try:
            import pytesseract
            from PIL import Image
           
            # Open and process image
            image = Image.open(file_path)
           
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
           
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
           
            return text.strip(), {
                'image_size': image.size,
                'image_mode': image.mode,
                'method': 'tesseract_ocr'
            }
           
        except ImportError:
            return "OCR libraries not available. Install pytesseract and Pillow.", {
                'error': 'OCR dependencies missing'
            }
        except Exception as e:
            return f"OCR processing failed: {str(e)}", {
                'error': str(e),
                'method': 'tesseract_ocr'
            }
   
    def _extract_from_docx(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract text from Word DOCX files."""
        try:
            import docx
           
            doc = docx.Document(file_path)
            text = ""
           
            # Extract paragraph text
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
           
            # Extract table text
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
           
            return text.strip(), {
                'paragraphs': len(doc.paragraphs),
                'tables': len(doc.tables),
                'method': 'python-docx'
            }
           
        except ImportError:
            return "Word document processing library not available. Install python-docx.", {
                'error': 'python-docx dependency missing'
            }
        except Exception as e:
            return f"Word document processing failed: {str(e)}", {
                'error': str(e)
            }
   
    def _extract_from_doc(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract text from legacy Word DOC files."""
        try:
            import textract
            text = textract.process(file_path).decode('utf-8')
            return text.strip(), {'method': 'textract'}
        except ImportError:
            return "Legacy Word document processing not available. Install textract.", {
                'error': 'textract dependency missing'
            }
        except Exception as e:
            return f"Legacy Word document processing failed: {str(e)}", {
                'error': str(e)
            }
   
    def _extract_from_text(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract text from plain text files."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
           
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    return text, {
                        'encoding': encoding,
                        'lines': len(text.split('\n')),
                        'method': 'direct_text'
                    }
                except UnicodeDecodeError:
                    continue
           
            return "Could not decode text file with common encodings", {
                'error': 'Encoding not supported'
            }
           
        except Exception as e:
            return f"Text processing failed: {str(e)}", {'error': str(e)}
   
    def _extract_from_json(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract and format JSON data."""
        try:
            data = json.loads(file_content.decode('utf-8'))
           
            # Convert JSON to readable text
            text = self._json_to_text(data)
           
            return text, {
                'json_keys': list(data.keys()) if isinstance(data, dict) else [],
                'json_type': type(data).__name__,
                'method': 'json_parser'
            }
           
        except json.JSONDecodeError as e:
            return f"Invalid JSON format: {str(e)}", {'error': 'Invalid JSON'}
        except Exception as e:
            return f"JSON processing failed: {str(e)}", {'error': str(e)}
   
    def _extract_from_xml(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract and format XML data."""
        try:
            root = ET.fromstring(file_content.decode('utf-8'))
           
            # Convert XML to readable text
            text = self._xml_to_text(root)
           
            return text, {
                'root_tag': root.tag,
                'xml_elements': len(list(root.iter())),
                'method': 'xml_parser'
            }
           
        except ET.ParseError as e:
            return f"Invalid XML format: {str(e)}", {'error': 'Invalid XML'}
        except Exception as e:
            return f"XML processing failed: {str(e)}", {'error': str(e)}
   
    def _extract_from_csv(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract and format CSV data."""
        try:
            text_content = file_content.decode('utf-8')
            reader = csv.reader(text_content.splitlines())
           
            rows = list(reader)
            if not rows:
                return "Empty CSV file", {'rows': 0}
           
            # Format as readable text
            text = ""
            headers = rows[0] if rows else []
           
            text += "CSV Headers: " + ", ".join(headers) + "\n\n"
           
            for i, row in enumerate(rows[1:], 1):
                text += f"Row {i}:\n"
                for header, value in zip(headers, row):
                    text += f"  {header}: {value}\n"
                text += "\n"
           
            return text.strip(), {
                'rows': len(rows),
                'columns': len(headers),
                'headers': headers,
                'method': 'csv_parser'
            }
           
        except Exception as e:
            return f"CSV processing failed: {str(e)}", {'error': str(e)}
   
    def _extract_from_excel(self, file_path: str, file_content: bytes) -> Tuple[str, Dict]:
        """Extract and format Excel data."""
        try:
            import pandas as pd
           
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
           
            text = ""
            total_rows = 0
           
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
               
                text += f"Sheet: {sheet_name}\n"
                text += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
               
                # Add first few rows as sample
                for idx, row in df.head(10).iterrows():
                    text += f"Row {idx + 1}:\n"
                    for col, value in row.items():
                        text += f"  {col}: {value}\n"
                    text += "\n"
               
                total_rows += len(df)
                text += f"... ({len(df)} total rows in this sheet)\n\n"
           
            return text.strip(), {
                'sheets': sheet_names,
                'total_sheets': len(sheet_names),
                'total_rows': total_rows,
                'method': 'pandas_excel'
            }
           
        except ImportError:
            return "Excel processing library not available. Install pandas and openpyxl.", {
                'error': 'pandas/openpyxl dependency missing'
            }
        except Exception as e:
            return f"Excel processing failed: {str(e)}", {'error': str(e)}
   
    def _json_to_text(self, data, indent=0) -> str:
        """Convert JSON data to readable text format."""
        text = ""
        prefix = "  " * indent
       
        if isinstance(data, dict):
            for key, value in data.items():
                text += f"{prefix}{key}: "
                if isinstance(value, (dict, list)):
                    text += "\n" + self._json_to_text(value, indent + 1)
                else:
                    text += f"{value}\n"
        elif isinstance(data, list):
            for i, item in enumerate(data):
                text += f"{prefix}[{i}]: "
                if isinstance(item, (dict, list)):
                    text += "\n" + self._json_to_text(item, indent + 1)
                else:
                    text += f"{item}\n"
        else:
            text += f"{prefix}{data}\n"
       
        return text
   
    def _xml_to_text(self, element, indent=0) -> str:
        """Convert XML element to readable text format."""
        text = ""
        prefix = "  " * indent
       
        text += f"{prefix}{element.tag}"
        if element.text and element.text.strip():
            text += f": {element.text.strip()}"
       
        # Add attributes
        if element.attrib:
            attrs = ", ".join([f"{k}={v}" for k, v in element.attrib.items()])
            text += f" ({attrs})"
       
        text += "\n"
       
        # Add children
        for child in element:
            text += self._xml_to_text(child, indent + 1)
       
        return text




def detect_file_type(filename: str) -> str:
    """Detect file type from filename."""
    return Path(filename).suffix.lower().lstrip('.')




def get_supported_formats() -> Dict[str, str]:
    """Get dictionary of supported formats and their descriptions."""
    return {
        'pdf': 'PDF Documents',
        'png': 'PNG Images',
        'jpg': 'JPEG Images',
        'jpeg': 'JPEG Images',
        'tiff': 'TIFF Images',
        'bmp': 'Bitmap Images',
        'docx': 'Word Documents (DOCX)',
        'doc': 'Word Documents (DOC)',
        'txt': 'Plain Text Files',
        'json': 'JSON Data Files',
        'xml': 'XML Data Files',
        'csv': 'CSV Spreadsheets',
        'xlsx': 'Excel Spreadsheets (XLSX)',
        'xls': 'Excel Spreadsheets (XLS)'
    }
    