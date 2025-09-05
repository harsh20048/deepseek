#!/usr/bin/env python3
"""
DeepSeek PDF Processor - Enhanced Command Line Tool

Extract structured data from German PDF documents using local DeepSeek model.

Usage Examples:
    python test.py                                    # Process default PDF
    python test.py -p invoice.pdf                     # Process specific PDF
    python test.py -p quote.pdf -o results.json       # Save results to file
    python test.py --model /path/to/model             # Use custom model path
    python test.py --verbose                          # Enable detailed logging
    python test.py --fields-only                      # Extract only fields
    python test.py --table-only                       # Extract only tables

Environment Variables:
    DEEPSEEK_MODEL_PATH: Path to DeepSeek model directory

Features:
    - Regex fallback for robust field extraction
    - Direct table extraction with pdfplumber
    - Configurable model path and output
    - Comprehensive error handling
    - Privacy-first: 100% offline processing
"""

import pdfplumber
import json
from tabulate import tabulate
import os
from datetime import datetime
import sys
import argparse
import logging
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# 0. Set environment variables for offline operation
# ---------------------------
os.environ["HF_HUB_OFFLINE"] = "1"  # Force HuggingFace to work offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force transformers to work offline
os.environ["HF_DATASETS_OFFLINE"] = "1"  # Force datasets to work offline
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "dummy_token"  # Bypass token requirement for local models
os.environ["OPENAI_API_KEY"] = "dummy_key"  # Bypass OpenAI API key requirement

# ---------------------------
# 0.0. Color formatting for terminal output
# ---------------------------
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_progress(text):
    """Print progress message"""
    print(f"{Colors.OKCYAN}🔄 {text}{Colors.ENDC}")

# ---------------------------
# Regex-based fallback extractor
# ---------------------------
def extract_fields_with_regex(text):
    """
    Fallback extractor using regex patterns for German documents
    Returns a dictionary with extracted fields or empty strings
    """
    print_info("Using regex fallback extractor...")
    
    result = {
        "Date": "",
        "Angebot": "",
        "SenderCompany": "",
        "SenderAddress": ""
    }
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Extract date (German format: DD.MM.YYYY or DD/MM/YYYY)
    date_patterns = [
        r'\b(\d{1,2}[./]\d{1,2}[./]\d{4})\b',
        r'Datum[:\s]*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'vom[:\s]*(\d{1,2}[./]\d{1,2}[./]\d{4})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["Date"] = match.group(1)
            break
    
    # Extract quotation number
    angebot_patterns = [
        r'Angebot(?:s?-?Nr\.?|s?nummer)?[:\s]*([A-Za-z0-9.\-/]+)',
        r'Offerte[:\s]*([A-Za-z0-9.\-/]+)',
        r'Quotation[:\s]*([A-Za-z0-9.\-/]+)',
        r'Nr\.?\s*([A-Za-z0-9.\-/]{3,})',
    ]
    
    for pattern in angebot_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Clean up the match
            angebot = match.group(1).strip()
            if len(angebot) >= 3:  # Minimum reasonable length
                result["Angebot"] = angebot
                break
    
    # Extract company name (look for legal forms)
    company_patterns = [
        r'([^.\n]+(?:GmbH|AG|UG|KG|OHG|mbH)(?:\s*&\s*Co\.?\s*KG)?)',
        r'([A-Z][^.\n]*(?:GmbH|AG|UG|KG|OHG|mbH))',
        r'([A-ZÄÖÜ][a-zäöüß\s]+(?:GmbH|AG|UG|KG|OHG|mbH))',
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the first reasonable match
            company = matches[0].strip()
            if len(company) > 5 and not any(word in company.lower() for word in ['seite', 'page', 'datum']):
                result["SenderCompany"] = company
                break
    
    # Extract address (look for postal codes and cities)
    address_patterns = [
        r'(\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)',  # German postal code + city
        r'([A-ZÄÖÜ][a-zäöüß\s]+-?[Ss]tr(?:aße|\.)\s*\d+[a-z]?(?:,?\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)?)',
        r'([A-ZÄÖÜ][a-zäöüß\s]+-?[Ww]eg\s*\d+[a-z]?(?:,?\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)?)',
    ]
    
    for pattern in address_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Combine street and city if found separately
            addresses = []
            for match in matches[:2]:  # Take first 2 matches
                if match.strip() and len(match.strip()) > 5:
                    addresses.append(match.strip())
            
            if addresses:
                result["SenderAddress"] = ", ".join(addresses)
                break
    
    return result

def preprocess_pdf_text(text, max_chars=2000):
    """
    Preprocess PDF text for better model performance
    Focus on first part which typically contains key information
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove multiple newlines
    text = re.sub(r'\s+', ' ', text)       # Normalize spaces
    text = text.strip()
    
    # Take first portion which usually contains header information
    if len(text) > max_chars:
        # Try to cut at a reasonable boundary (sentence or paragraph)
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        cut_point = max(last_period, last_newline)
        if cut_point > max_chars * 0.8:  # If we found a good cut point
            text = text[:cut_point + 1]
        else:
            text = truncated + "..."
    
    return text

def extract_tables_with_pdfplumber(pdf_path):
    """
    Extract tables directly using pdfplumber without AI model
    Returns a list of normalized table data
    """
    print_progress("Extracting tables directly from PDF...")
    
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                
                for table_num, table in enumerate(page_tables):
                    if table and len(table) > 1:  # Skip empty or single-row tables
                        # Normalize table data
                        normalized_table = []
                        headers = None
                        
                        for row_idx, row in enumerate(table):
                            # Clean up row data
                            cleaned_row = []
                            for cell in row:
                                if cell is None:
                                    cleaned_row.append("")
                                else:
                                    # Clean whitespace and normalize
                                    cleaned_cell = str(cell).strip().replace('\n', ' ')
                                    cleaned_row.append(cleaned_cell)
                            
                            # First non-empty row becomes headers
                            if headers is None and any(cell for cell in cleaned_row):
                                headers = cleaned_row
                                continue
                            
                            # Skip rows that look like totals or summaries
                            row_text = ' '.join(cleaned_row).lower()
                            if any(keyword in row_text for keyword in ['summe', 'total', 'gesamt', 'zwischensumme']):
                                continue
                            
                            # Skip empty rows
                            if not any(cell.strip() for cell in cleaned_row):
                                continue
                            
                            # Create row dictionary
                            if headers:
                                row_dict = {}
                                for i, header in enumerate(headers):
                                    if i < len(cleaned_row):
                                        # Truncate long values
                                        value = cleaned_row[i][:40] + "..." if len(cleaned_row[i]) > 40 else cleaned_row[i]
                                        row_dict[header or f"Column_{i+1}"] = value
                                    else:
                                        row_dict[header or f"Column_{i+1}"] = ""
                                
                                normalized_table.append(row_dict)
                        
                        if normalized_table:
                            tables.extend(normalized_table)
                            print_info(f"Extracted {len(normalized_table)} rows from page {page_num}, table {table_num + 1}")
    
    except Exception as e:
        print_error(f"Error extracting tables with pdfplumber: {e}")
        return []
    
    return tables

# ---------------------------
# 1. Extract raw text from PDF
# ---------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with progress indication"""
    print_progress(f"Extracting text from PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print_error(f"PDF file not found: {pdf_path}")
        return None
    
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print_info(f"Processing {total_pages} pages...")
            
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                print_progress(f"Processed page {i}/{total_pages}")
            
            print_success(f"Successfully extracted text from {total_pages} pages")
            print_info(f"Total characters extracted: {len(text)}")
            
    except Exception as e:
        print_error(f"Error extracting text from PDF: {e}")
        return None
    
    return text

# ---------------------------
# 2. Parse command line arguments and configure paths
# ---------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="DeepSeek PDF Processor - Extract structured data from German PDFs")
    parser.add_argument("--pdf", default="german_quotation.pdf", help="Path to PDF file to process")
    parser.add_argument("--model", default=None, help="Path to DeepSeek model directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

args = parse_arguments()

# Configure logging
if args.verbose:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deepseek_pdf_processor.log'),
            logging.StreamHandler()
        ]
    )
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

print_header("DEEPSEEK PDF PROCESSING SYSTEM")
print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_info(f"Processing PDF: {args.pdf}")
logger.info(f"Starting processing of {args.pdf}")

# Configure model path
if args.model:
    model_path = args.model
else:
    # Try environment variable first, then default to local model directory
    model_path = os.environ.get("DEEPSEEK_MODEL_PATH", 
                               os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

print_header("MODEL INITIALIZATION")
print_info(f"Loading DeepSeek model from local path: {model_path}")
print_warning("Running in OFFLINE mode - no data will be sent to external servers")


print_progress("Initializing DeepSeek model...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print_progress("Loading local DeepSeek model...")
    model_path = "C:/Users/HARSH/OneDrive/Desktop/pdf/model"
    
    # Load the model and tokenizer locally
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )
    
    print_success("DeepSeek model loaded successfully in offline mode")
    print_success("Data privacy: All processing is done locally, no data sent to external servers")
except Exception as e:
    print_error(f"Error loading DeepSeek model: {e}")
    print_warning("Make sure all model files (config.json, tokenizer.json, etc.) are in the model directory")
    sys.exit(1)

# ---------------------------
# 3. Load PDF and process with local model
# ---------------------------
print_header("PDF PROCESSING")

# Check if PDF file exists
if not os.path.exists(args.pdf):
    print_error(f"PDF file not found: {args.pdf}")
    print_info("Usage: python test.py --pdf path/to/your/file.pdf")
    sys.exit(1)

pdf_text = extract_text_from_pdf(args.pdf)

if pdf_text is None:
    print_error("Failed to extract text from PDF. Exiting...")
    sys.exit(1)

print_success("PDF content extracted successfully")

def preprocess_pdf_text(text, max_chars=1500):
    """Preprocess PDF text for better model performance"""
    if not text:
        return ""
    
    # Normalize whitespace
    import re
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Take first portion which usually contains key information
    if len(text) > max_chars:
        # Try to cut at a sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.7:  # If we find a period in the last 30%
            text = truncated[:last_period + 1]
        else:
            text = truncated + "..."
        print_info(f"Text truncated to {len(text)} characters for better processing")
    
    return text

def extract_fields_with_regex(text):
    """Enhanced regex-based field extractor with broader patterns"""
    import re
    
    fields = {
        "Date": "",
        "Angebot": "",
        "SenderCompany": "",
        "SenderAddress": ""
    }
    
    # Clean text for better pattern matching
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    
    # 1. Extract date - support multiple formats
    date_patterns = [
        r'\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b',  # DD.MM.YYYY or DD/MM/YY
        r'\b(\d{4}-\d{2}-\d{2})\b',              # YYYY-MM-DD
        r'(?:Datum|vom|am)[:.]?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})',  # "Datum: DD.MM.YYYY"
    ]
    
    for pattern in date_patterns:
        date_match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if date_match:
            date_str = date_match.group(1)
            # Skip placeholder dates
            if not re.match(r'XX\.XX\.20XX|00\.00\.0000', date_str):
                fields["Date"] = date_str
                break
    
    # 2. Extract quotation number - recognize multiple labels
    angebot_patterns = [
        r'(Angebot(?:s|s?-?Nr\.?|s?nummer)?)\s*[:.-]?\s*([A-Za-z0-9./-]+)',
        r'(Belegnummer|Beleg-Nr\.?)\s*[:.-]?\s*([A-Za-z0-9./-]+)',
        r'(ANG|AG)[-.]?\s*(\d{4}[-.]?\d{1,4})',  # ANG-2025-330
        r'(Offerte|Kostenvoranschlag)[-.]?(?:Nr\.?)?\s*[:.-]?\s*([A-Za-z0-9./-]+)',
    ]
    
    for pattern in angebot_patterns:
        angebot_match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if angebot_match:
            fields["Angebot"] = angebot_match.group(2).strip()
            break
    
    # 3. Extract company name - look in first 10 lines
    lines = cleaned_text.split('\n')[:10]
    company_patterns = [
        r'([A-Za-zÄÖÜäöüß\s&.-]{3,50}(?:GmbH|AG|KG|OHG|mbH|e\.K\.|UG))',
        r'([A-Za-zÄÖÜäöüß\s&.-]{10,50})\s+(?:Straße|Str\.|Platz)',  # Company before address
    ]
    
    for line in lines:
        for pattern in company_patterns:
            company_match = re.search(pattern, line)
            if company_match:
                company = company_match.group(1).strip()
                if len(company) > 5:  # Avoid single words
                    fields["SenderCompany"] = company
                    break
        if fields["SenderCompany"]:
            break
    
    # 4. Extract address - look for postal code and city
    address_patterns = [
        r'([A-Za-zÄÖÜäöüß\s.-]+\d{5}\s+[A-Za-zÄÖÜäöüß\s-]+)',  # Street + PLZ + City
        r'(\d{5}\s+[A-Za-zÄÖÜäöüß\s-]+)',  # PLZ + City only
        r'([A-Za-zÄÖÜäöüß\s.-]+(?:straße|str\.|platz|weg|gasse)[^0-9]*\d+[^0-9]*\d{5}\s+[A-Za-zÄÖÜäöüß\s-]+)',  # Full address
    ]
    
    for pattern in address_patterns:
        address_match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if address_match:
            address = address_match.group(1).strip()
            # Clean up the address
            address = re.sub(r'\s+', ' ', address)
            if len(address) > 10:  # Meaningful address
                fields["SenderAddress"] = address
                break
    
    return fields

def extract_table_with_pdfplumber(pdf_path):
    """Extract table data directly using pdfplumber"""
    import pdfplumber
    
    print_progress("Extracting tables directly from PDF...")
    
    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                
                for table in tables:
                    if not table or len(table) < 2:  # Skip empty or single-row tables
                        continue
                    
                    # Process header row
                    header = table[0] if table else []
                    header = [str(cell).strip() if cell else f"Column_{i+1}" for i, cell in enumerate(header)]
                    
                    # Process data rows
                    for row in table[1:]:
                        if not row or all(not cell or str(cell).strip() == '' for cell in row):
                            continue  # Skip empty rows
                        
                        # Skip total/summary rows
                        first_cell = str(row[0] or '').lower().strip()
                        if any(keyword in first_cell for keyword in ['summe', 'total', 'gesamt', 'netto', 'brutto', 'mwst']):
                            continue
                        
                        # Create row dict
                        row_dict = {}
                        for i, cell in enumerate(row):
                            col_name = header[i] if i < len(header) else f"Column_{i+1}"
                            cell_value = str(cell).strip() if cell else ""
                            
                            # Truncate to 40 characters
                            if len(cell_value) > 40:
                                cell_value = cell_value[:40] + "..."
                            
                            row_dict[col_name] = cell_value
                        
                        all_tables.append(row_dict)
                
                if all_tables:  # Found tables, don't need to check more pages
                    break
    
    except Exception as e:
        print_error(f"Error extracting tables with pdfplumber: {e}")
        return []
    
    print_success(f"Extracted {len(all_tables)} table rows using pdfplumber")
    return all_tables

def query_local_model(prompt, max_new_tokens=512):
    """Query the local DeepSeek model with improved parameters"""
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Check if input is too long and truncate if necessary
        max_input_length = 2048  # Set a reasonable input limit
        if inputs.shape[1] > max_input_length:
            inputs = inputs[:, -max_input_length:]  # Take the last max_input_length tokens
            print_warning(f"Input truncated to {max_input_length} tokens")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=max_new_tokens,  # Control only the generated output length
                do_sample=False,                # Deterministic output for reproducibility
                temperature=0.0,                # No randomness for consistent results
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        # Decode only the new tokens (excluding the input prompt)
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        print_error(f"Error querying model: {e}")
        return "Error processing query"

# ---------------------------
# 4. Enhanced field extraction with validation and fallback
# ---------------------------
print_header("FIELD EXTRACTION")
print_progress("Extracting structured fields from German quotation...")

# Preprocess PDF text for better model performance
processed_text = preprocess_pdf_text(pdf_text, max_chars=2500)  # Increased limit

# Improved prompt with explicit instructions
query = """
Extract the following fields from this German quotation PDF and return them as strict JSON with the keys Date, Angebot, SenderCompany, SenderAddress.

Fields to extract:
- Date: Document date (format DD.MM.YYYY or DD/MM/YYYY)
- Angebot: Quotation/offer number (may be labeled as Angebot, Angebotsnummer, Belegnummer, ANG, Nr., etc.)
- SenderCompany: Company name of the sender (look for GmbH, AG, UG, KG, etc.)
- SenderAddress: Complete address of the sender (street, postal code, city)

CRITICAL RULES:
1. Return ONLY a valid JSON object, no additional text
2. If a field is missing, set it to an empty string ""
3. Do not include any additional keys or explanatory text
4. Use exactly these key names: Date, Angebot, SenderCompany, SenderAddress
5. Do not include bullet points, markdown, or formatting
6. The response must be parseable by json.loads()

JSON format:
{
  "Date": "",
  "Angebot": "",
  "SenderCompany": "",
  "SenderAddress": ""
}
"""

print_info("Querying DeepSeek model for field extraction...")
full_prompt = f"PDF Content:\n{processed_text}\n\n{query}"
fields_response = query_local_model(full_prompt, max_new_tokens=256)

# Validate and parse JSON response
import json
parsed_fields = None
try:
    parsed_fields = json.loads(fields_response)
    # Validate that we got reasonable fields
    if not isinstance(parsed_fields, dict):
        raise ValueError("Response is not a dictionary")
    
    # Check if all required keys exist and have meaningful values
    required_keys = ["Date", "Angebot", "SenderCompany", "SenderAddress"]
    if not all(k in parsed_fields for k in required_keys):
        raise ValueError("Missing required keys")
    
    # Check if we got meaningful data (not all empty)
    if all(not parsed_fields.get(k, "").strip() for k in required_keys):
        raise ValueError("All fields are empty")
        
    print_success("AI model successfully extracted structured fields")
    
except (json.JSONDecodeError, ValueError) as e:
    print_warning(f"AI model returned invalid response: {e}")
    print_info("Falling back to regex-based extraction...")
    parsed_fields = extract_fields_with_regex(pdf_text)
    print_success("Regex fallback extraction completed")

print_header("EXTRACTED FIELDS")
print(f"{Colors.OKGREEN}📅 Date: {parsed_fields.get('Date', 'Not found')}{Colors.ENDC}")
print(f"{Colors.OKGREEN}📄 Angebot: {parsed_fields.get('Angebot', 'Not found')}{Colors.ENDC}")
print(f"{Colors.OKGREEN}🏢 Company: {parsed_fields.get('SenderCompany', 'Not found')}{Colors.ENDC}")
print(f"{Colors.OKGREEN}📍 Address: {parsed_fields.get('SenderAddress', 'Not found')}{Colors.ENDC}")

# Also show raw response for debugging
print(f"\n{Colors.OKCYAN}Raw AI Response:{Colors.ENDC}")
print(f"{Colors.OKCYAN}{fields_response}{Colors.ENDC}")

# ---------------------------
# 5. Enhanced table extraction with pdfplumber fallback
# ---------------------------
print_header("TABLE EXTRACTION")

# Try pdfplumber first (more reliable for structured tables)
pdfplumber_tables = extract_tables_with_pdfplumber(args.pdf)

cleaned_table = []
if pdfplumber_tables:
    print_success("Using pdfplumber table extraction (more reliable)")
    cleaned_table = pdfplumber_tables
else:
    print_info("pdfplumber found no tables, trying AI model extraction...")

table_query = """
Extract all tabular data from this German quotation and return as a JSON array.
Columns may include:
- Pos/Artikel-Nr: Position or article number
- Bezeichnung: Description of item/service
- Menge: Quantity
- Einzelpreis: Unit price
- Gesamt: Total price

Rules:
1. Return ONLY a valid JSON array of objects
2. Each object represents one table row
3. Use consistent key names across all rows
4. Truncate values to max 40 characters
5. Skip header rows and total/summary rows
6. Only include actual product/service items

Example format:
[
  {"Pos": "1", "Bezeichnung": "Product A", "Menge": "2", "Einzelpreis": "10.00", "Gesamt": "20.00"},
  {"Pos": "2", "Bezeichnung": "Product B", "Menge": "1", "Einzelpreis": "15.00", "Gesamt": "15.00"}
]
"""

print_info("Querying DeepSeek model for table extraction...")
    full_table_prompt = f"PDF Content:\n{processed_text}\n\n{table_query}"
    table_response = query_local_model(full_table_prompt, max_new_tokens=1024)
    
    # Try to parse AI response
    try:
        ai_table = json.loads(table_response)
        if isinstance(ai_table, list) and ai_table:
            print_success("AI model successfully extracted table data")
            cleaned_table = ai_table
        else:
            raise ValueError("AI returned empty or invalid table")
    except (json.JSONDecodeError, ValueError) as e:
        print_warning(f"AI model table extraction failed: {e}")
        print_info("No reliable table data could be extracted")
        cleaned_table = []

# ---------------------------
# 7. Print as pretty table
# ---------------------------
print_header("EXTRACTED TABLE DATA")

if cleaned_table:
    headers = cleaned_table[0].keys()
    rows = [row.values() for row in cleaned_table]
    
    print_success(f"Found {len(cleaned_table)} table rows")
    print(f"\n{Colors.OKCYAN}{tabulate(rows, headers=headers, tablefmt='grid')}{Colors.ENDC}")
else:
    print_warning("No table data found in the PDF")

# ---------------------------
# 8. Completion Summary
# ---------------------------
# ---------------------------
# Save results if output file specified
# ---------------------------
if hasattr(args, 'output') and args.output:
    try:
        output_data = {
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "pdf_file": args.pdf,
                "model_path": model_path,
                "processing_method": "DeepSeek + pdfplumber"
            },
            "extracted_fields": parsed_fields if 'parsed_fields' in locals() else {},
            "extracted_tables": cleaned_table if 'cleaned_table' in locals() else []
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print_success(f"Results saved to: {args.output}")
        
    except Exception as e:
        print_error(f"Failed to save results: {e}")

print_header("PROCESSING COMPLETE")
print_success(f"PDF processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_info("All data processed locally with DeepSeek model")
print_success("🔒 Data privacy maintained - no external data transmission")

# Summary statistics
if 'parsed_fields' in locals():
    fields_found = sum(1 for v in parsed_fields.values() if v and str(v).strip())
    print_info(f"Fields extracted: {fields_found}/4")

if 'cleaned_table' in locals():
    print_info(f"Table rows extracted: {len(cleaned_table)}")

print(f"\n{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
print(f"{Colors.OKGREEN}{'PDF PROCESSING SUCCESSFUL!'.center(80)}{Colors.ENDC}")
print(f"{Colors.OKGREEN}{'='*80}{Colors.ENDC}")