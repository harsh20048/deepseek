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

# Try to import pdfplumber with fallback
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("WARNING: pdfplumber not available. Table extraction will be limited.")

# ---------------------------
# 0. Set environment variables for offline operation
# ---------------------------
os.environ["HF_HUB_OFFLINE"] = "1"  # Force HuggingFace to work offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force transformers to work offline
os.environ["HF_DATASETS_OFFLINE"] = "1"  # Force datasets to work offline
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "dummy_token"  # Bypass token requirement for local models
os.environ["OPENAI_API_KEY"] = "dummy_key"  # Bypass OpenAI API key requirement

# ---------------------------
# Color formatting for terminal output
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
# Command line argument parsing
# ---------------------------
def setup_args():
    parser = argparse.ArgumentParser(description='DeepSeek PDF Processor - Extract structured data from German PDFs')
    parser.add_argument('--pdf', '-p', type=str, default="german_quotation.pdf",
                       help='Path to the PDF file to process (default: german_quotation.pdf)')
    parser.add_argument('--model', '-m', type=str, 
                       default=os.environ.get('DEEPSEEK_MODEL_PATH', 
                               os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')),
                       help='Path to DeepSeek model directory (default: ./model or DEEPSEEK_MODEL_PATH env var)')
    parser.add_argument('--output', '-o', type=str, 
                       help='Output JSON file path (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--table-only', action='store_true',
                       help='Extract only table data, skip field extraction')
    parser.add_argument('--fields-only', action='store_true',
                       help='Extract only fields, skip table extraction')
    return parser.parse_args()

# ---------------------------
# Text preprocessing and extraction functions
# ---------------------------
def preprocess_pdf_text(text, max_chars=2500):
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

def extract_tables_with_pdfplumber(pdf_path):
    """
    Extract tables directly using pdfplumber without AI model
    Returns a list of normalized table data
    """
    if not PDFPLUMBER_AVAILABLE:
        print_warning("pdfplumber not available, skipping table extraction")
        return []
        
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

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with progress indication"""
    if not PDFPLUMBER_AVAILABLE:
        print_error("pdfplumber not available, cannot extract text from PDF")
        return None
        
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
                top_p=1.0,                      # Explicitly set for greedy decoding
                top_k=0,                        # Explicitly disable top-k for greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
        
    except Exception as e:
        print_error(f"Error querying model: {e}")
        return "Error processing query"

def truncate_table_json(table_json, max_len=40):
    """Clean and truncate table data for better display"""
    try:
        data = json.loads(table_json)
        for row in data:
            for key, value in row.items():
                if isinstance(value, str) and len(value) > max_len:
                    row[key] = value[:max_len] + "..."
        return data
    except Exception as e:
        print_error(f"Error cleaning table JSON: {e}")
        return []

# ---------------------------
# Main execution
# ---------------------------
def main():
    global tokenizer, model
    
    # Parse command line arguments
    args = setup_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate arguments
    if args.fields_only and args.table_only:
        print_error("Cannot specify both --fields-only and --table-only")
        sys.exit(1)
    
    print_header("DEEPSEEK PDF PROCESSING SYSTEM")
    print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ---------------------------
    # Model initialization
    # ---------------------------
    print_header("MODEL INITIALIZATION")
    model_path = args.model  # Use the resolved model path from arguments
    print_info(f"Loading DeepSeek model from local path: {model_path}")
    print_warning("Running in OFFLINE mode - no data will be sent to external servers")
    
    print_progress("Initializing DeepSeek model...")
    try:
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
        print_warning("Make sure all model files are in the model directory")
        sys.exit(1)
    
    # ---------------------------
    # PDF text extraction
    # ---------------------------
    print_header("PDF PROCESSING")
    pdf_text = extract_text_from_pdf(args.pdf)
    
    if pdf_text is None:
        print_error("Failed to extract text from PDF. Exiting...")
        sys.exit(1)
    
    # Preprocess PDF text for better model performance
    processed_text = preprocess_pdf_text(pdf_text, max_chars=2500)
    
    extracted_fields = {}
    extracted_tables = []
    
    # ---------------------------
    # Field extraction (unless table-only mode)
    # ---------------------------
    if not args.table_only:
        print_header("FIELD EXTRACTION")
        print_progress("Extracting structured fields from German quotation...")
        
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
        
        # Validate and parse JSON response with fallback
        parsed_fields = None
        try:
            parsed_fields = json.loads(fields_response)
            
            # Validate that we have the expected keys
            required_keys = {"Date", "Angebot", "SenderCompany", "SenderAddress"}
            if not isinstance(parsed_fields, dict) or not all(k in parsed_fields for k in required_keys):
                print_warning("AI model returned incomplete JSON, using fallback extractor...")
                parsed_fields = None
                
        except json.JSONDecodeError as e:
            print_warning(f"AI model returned invalid JSON: {e}")
            print_warning("Using regex fallback extractor...")
        
        # Use regex fallback if AI failed
        if parsed_fields is None:
            parsed_fields = extract_fields_with_regex(processed_text)
        
        extracted_fields = parsed_fields
        
        print_header("EXTRACTED FIELDS")
        if parsed_fields:
            print(f"{Colors.OKGREEN}📋 Document Information:{Colors.ENDC}")
            print(f"{Colors.OKBLUE}📅 Date: {Colors.ENDC}{parsed_fields.get('Date', 'Not found')}")
            print(f"{Colors.OKBLUE}📄 Quotation: {Colors.ENDC}{parsed_fields.get('Angebot', 'Not found')}")
            print(f"{Colors.OKBLUE}🏢 Company: {Colors.ENDC}{parsed_fields.get('SenderCompany', 'Not found')}")
            print(f"{Colors.OKBLUE}📍 Address: {Colors.ENDC}{parsed_fields.get('SenderAddress', 'Not found')}")
        else:
            print_error("Failed to extract fields from document")
    
    # ---------------------------
    # Table extraction (unless fields-only mode)
    # ---------------------------
    if not args.fields_only:
        print_header("TABLE EXTRACTION")
        
        # Try pdfplumber first (more reliable for structured tables)
        pdfplumber_tables = extract_tables_with_pdfplumber(args.pdf)
        
        if pdfplumber_tables:
            print_success("Using pdfplumber table extraction (more reliable)")
            extracted_tables = pdfplumber_tables
        else:
            print_info("pdfplumber found no tables, trying AI model extraction...")
            
            table_query = """
Extract all tabular data from this German quotation.
Columns may include:
- Artikel-Nr (or Pos, Artikelnummer, etc.)
- Bezeichnung (description)
- Menge (quantity)
- Preis (Einzelpreis, unit price)
- Gesamt (Gesamtpreis, total)

Rules:
1. Return result as JSON array of rows.
2. Each value must be plain text.
3. Each value must be max 40 characters (truncate if longer).
4. Only include relevant product/item rows (no headers, no totals).
"""
            
            print_info("Querying DeepSeek model for table extraction...")
            full_table_prompt = f"PDF Content:\n{processed_text}\n\n{table_query}"
            table_response = query_local_model(full_table_prompt, max_new_tokens=1024)
            
            # Process and display table
            cleaned_table = truncate_table_json(table_response)
            extracted_tables = cleaned_table
        
        # Display table results
        print_header("EXTRACTED TABLE DATA")
        if extracted_tables:
            print_success(f"Found {len(extracted_tables)} table rows")
            print(f"\n{Colors.OKCYAN}{tabulate(extracted_tables, headers='keys', tablefmt='grid')}{Colors.ENDC}")
        else:
            print_warning("No table data found in the PDF")
    
    # ---------------------------
    # Save results if output file specified
    # ---------------------------
    if args.output:
        try:
            output_data = {
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "pdf_file": args.pdf,
                    "model_path": args.model,
                    "processing_method": "DeepSeek + pdfplumber"
                },
                "extracted_fields": extracted_fields,
                "extracted_tables": extracted_tables
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print_success(f"Results saved to: {args.output}")
            
        except Exception as e:
            print_error(f"Failed to save results: {e}")
    
    # ---------------------------
    # Completion summary
    # ---------------------------
    print_header("PROCESSING COMPLETE")
    print_success(f"PDF processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info("All data processed locally with DeepSeek model")
    print_success("🔒 Data privacy maintained - no external data transmission")
    
    # Summary statistics
    if extracted_fields:
        fields_found = sum(1 for v in extracted_fields.values() if v and str(v).strip())
        print_info(f"Fields extracted: {fields_found}/4")
    
    if extracted_tables:
        print_info(f"Table rows extracted: {len(extracted_tables)}")
    
    print(f"\n{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{'PDF PROCESSING SUCCESSFUL!'.center(80)}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{'='*80}{Colors.ENDC}")

if __name__ == "__main__":
    main()
