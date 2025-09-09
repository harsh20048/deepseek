#!/usr/bin/env python3
"""
DeepSeek PDF Processor - Enhanced Command Line Tool

Extract structured data from German PDF documents using local DeepSeek model.

REQUIREMENTS:
- Install dependencies: pip install -r requirements.txt
- Ensure accelerate package is installed for GPU support: pip install accelerate

NEW FEATURES:
1. Intelligent Rolls-Royce Filtering
   - Automatically detects and filters out Rolls-Royce Power Systems AG company names
   - Filters out Rolls-Royce addresses (Maybachplatz 1, 88045 Friedrichshafen, Germany)
   - Uses DeepSeek AI for intelligent detection with regex fallback
   - Use --no-filter flag to disable filtering for testing

2. Batch Processing (NEW)
   - Process all PDF files in a specified folder automatically
   - Support for parallel processing to speed up batch operations
   - Comprehensive output with individual results and summary
   - Automatic Rolls-Royce filtering for all files in batch

Example usage:
  python test.py                                      # Process all PDFs in default folder (batch mode)
  python test.py --pdf document.pdf                   # Process single PDF file
  python test.py --pdf document.pdf --no-filter       # Without filtering
  python test.py --pdf document.pdf --fields-only     # Extract only fields
  python test.py --pdf document.pdf --table-only      # Extract only tables
  python test.py --pdf document.pdf --compact-tables  # Display tables in compact format
  python test.py --test-filter                        # Test filtering functionality
  
  # Batch processing (NEW FEATURE)
  python test.py --folder /path/to/pdfs               # Process all PDFs in folder (3 parallel by default)
  python test.py --folder /path/to/pdfs --parallel 4  # Use 4 parallel processes
  python test.py --folder /path/to/pdfs --fast        # Fast mode (4 parallel processes)
  python test.py --folder /path/to/pdfs --slow        # Slow mode (1 parallel process)
  python test.py --folder /path/to/pdfs --turbo       # Turbo mode (6 parallel processes)
  python test.py --folder /path/to/pdfs -o ./results  # Custom output directory
  python test.py --folder /path/to/pdfs --fields-only # Extract only fields from all PDFs
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
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Try to import pdfplumber with fallback
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("WARNING: pdfplumber not available. Table extraction will be limited.")

# ---------------------------
# 0. Environment variables for offline mode
# ---------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "dummy_token"
os.environ["OPENAI_API_KEY"] = "dummy_key"

# ---------------------------
# Default paths and settings (change here if needed)
# ---------------------------
DEFAULT_PDF = r"C:\Users\HARSH\OneDrive\Desktop\project\secure_pdf_extractor\pdfs"      # <-- PDF folder path
DEFAULT_MODEL = r"C:\Users\HARSH\OneDrive\Desktop\project\model"             # <-- Change your model path
DEFAULT_PARALLEL = 3                                                        # <-- Default parallel processes (EASY TO CHANGE!)

# ---------------------------
# Parallel Processing Configuration
# ---------------------------
# Quick presets for different system types:
# BUDGET_PC = 2      # For systems with 8GB RAM or less
# MID_RANGE = 3      # For systems with 16GB RAM (recommended)
# HIGH_END = 4       # For systems with 32GB+ RAM
# WORKSTATION = 6    # For powerful workstations

# Uncomment one of these to quickly change your default:
# DEFAULT_PARALLEL = 2  # Budget PC
# DEFAULT_PARALLEL = 3  # Mid-range PC (current setting)
# DEFAULT_PARALLEL = 4  # High-end PC
# DEFAULT_PARALLEL = 6  # Workstation

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
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

def print_success(text): print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")
def print_error(text): print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")
def print_info(text): print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")
def print_warning(text): print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")
def print_progress(text): print(f"{Colors.OKCYAN}🔄 {text}{Colors.ENDC}")

# ---------------------------
# Command line arguments
# ---------------------------
def setup_args():
    parser = argparse.ArgumentParser(description='DeepSeek PDF Processor')
    parser.add_argument('--pdf', type=str, default=DEFAULT_PDF,
                       help=f'Path to the PDF file or folder (default: {DEFAULT_PDF})')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                       help=f'Path to DeepSeek model directory (default: {DEFAULT_MODEL})')
    parser.add_argument('--output', type=str, help='Optional JSON output file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--table-only', action='store_true', help='Extract only tables')
    parser.add_argument('--fields-only', action='store_true', help='Extract only fields')
    parser.add_argument('--no-filter', action='store_true', help='Disable Rolls-Royce filtering (for testing)')
    parser.add_argument('--test-filter', action='store_true', help='Test Rolls-Royce filtering functionality')
    parser.add_argument('--folder', '-f', type=str, help='Process all PDFs in specified folder')
    parser.add_argument('--output-dir', '-o', type=str, default='./extracted_results', help='Output directory for batch processing results')
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL, help=f'Number of parallel processes for batch processing (default: {DEFAULT_PARALLEL})')
    parser.add_argument('--fast', action='store_true', help='Use fast processing (4 parallel processes)')
    parser.add_argument('--slow', action='store_true', help='Use slow processing (1 parallel process)')
    parser.add_argument('--turbo', action='store_true', help='Use turbo processing (6 parallel processes)')
    parser.add_argument('--compact-tables', action='store_true', help='Display tables in compact format')
    return parser.parse_args()

# ---------------------------
# PDF processing helpers
# ---------------------------
def preprocess_pdf_text(text, max_chars=2500):
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_chars:
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cut_point = max(last_period, last_newline)
        text = text[:cut_point + 1] if cut_point > max_chars * 0.8 else truncated + "..."
    return text

def is_rolls_royce_company(company_text):
    """Check if the company text matches Rolls-Royce Power Systems AG"""
    if not company_text:
        return False
    
    # Normalize text for comparison
    normalized = re.sub(r'\s+', ' ', company_text.strip().lower())
    
    # Check for various forms of Rolls-Royce Power Systems AG
    rolls_royce_patterns = [
        r'rolls[\s\-]?royce[\s\-]?power[\s\-]?systems[\s\-]?ag',
        r'rolls[\s\-]?royce[\s\-]?power[\s\-]?systems',
        r'rolls[\s\-]?royce[\s\-]?solutions[\s\-]?gmbh',
        r'rolls[\s\-]?royce[\s\-]?solutions',
    ]
    
    for pattern in rolls_royce_patterns:
        if re.search(pattern, normalized):
            return True
    
    return False

def is_rolls_royce_address(address_text):
    """Check if the address text matches the Rolls-Royce address"""
    if not address_text:
        return False
    
    # Normalize text for comparison
    normalized = re.sub(r'\s+', ' ', address_text.strip().lower())
    
    # Check for the specific Rolls-Royce address components
    address_indicators = [
        r'maybachplatz\s*1',
        r'88045\s*friedrichshafen',
        r'rolls[\s\-]?royce[\s\-]?power[\s\-]?systems',
        r'rolls[\s\-]?royce[\s\-]?solutions',
    ]
    
    # Count how many indicators are present
    matches = sum(1 for pattern in address_indicators if re.search(pattern, normalized))
    
    # If 2 or more indicators match, it's likely the Rolls-Royce address
    return matches >= 2

def filter_rolls_royce_content(text):
    """Use DeepSeek to intelligently detect and filter out Rolls-Royce content"""
    try:
        # Create a prompt to identify Rolls-Royce content
        detection_prompt = f"""
Analyze the following text and identify if it contains information about Rolls-Royce Power Systems AG or Rolls-Royce Solutions GmbH.

Text to analyze:
{text[:1000]}

Please respond with a JSON object indicating:
1. "contains_rolls_royce_company": true/false - if company name matches Rolls-Royce Power Systems AG or similar
2. "contains_rolls_royce_address": true/false - if address contains Maybachplatz 1, 88045 Friedrichshafen, Germany or similar
3. "confidence": 0.0-1.0 - confidence in the detection

Respond only with valid JSON:
"""
        
        # Query the model for intelligent detection
        response = query_local_model(detection_prompt, max_new_tokens=200)
        
        try:
            detection_result = json.loads(response)
            return detection_result
        except json.JSONDecodeError:
            # Fallback to regex-based detection
            return {
                "contains_rolls_royce_company": is_rolls_royce_company(text),
                "contains_rolls_royce_address": is_rolls_royce_address(text),
                "confidence": 0.5
            }
    except Exception as e:
        print_warning(f"DeepSeek detection failed, using regex fallback: {e}")
        # Fallback to regex-based detection
        return {
            "contains_rolls_royce_company": is_rolls_royce_company(text),
            "contains_rolls_royce_address": is_rolls_royce_address(text),
            "confidence": 0.3
        }

def extract_fields_with_regex(text):
    print_info("Using regex fallback extractor...")
    result = {"Date": "", "Angebot": "", "SenderCompany": "", "SenderAddress": ""}
    text = re.sub(r'\s+', ' ', text.strip())

    # Date
    for pattern in [
        r'\b(\d{1,2}[./]\d{1,2}[./]\d{4})\b',
        r'Datum[:\s]*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'vom[:\s]*(\d{1,2}[./]\d{1,2}[./]\d{4})',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["Date"] = match.group(1); break

    # Angebot
    for pattern in [
        r'Angebot(?:s?-?Nr\.?|s?nummer)?[:\s]*([A-Za-z0-9.\-/]+)',
        r'Offerte[:\s]*([A-Za-z0-9.\-/]+)',
        r'Quotation[:\s]*([A-Za-z0-9.\-/]+)',
        r'Nr\.?\s*([A-Za-z0-9.\-/]{3,})',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and len(match.group(1)) >= 3:
            result["Angebot"] = match.group(1).strip(); break

    # Company - with Rolls-Royce filtering
    for pattern in [
        r'([^.\n]+(?:GmbH|AG|UG|KG|OHG|mbH)(?:\s*&\s*Co\.?\s*KG)?)',
        r'([A-Z][^.\n]*(?:GmbH|AG|UG|KG|OHG|mbH))',
        r'([A-ZÄÖÜ][a-zäöüß\s]+(?:GmbH|AG|UG|KG|OHG|mbH))',
    ]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for company in matches:
                company = company.strip()
                if len(company) > 5 and not is_rolls_royce_company(company):
                    result["SenderCompany"] = company
                    print_info(f"Found company: {company}")
                    break
            if result["SenderCompany"]:
                break

    # Address - with Rolls-Royce filtering
    for pattern in [
        r'(\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)',
        r'([A-ZÄÖÜ][a-zäöüß\s]+-?[Ss]tr(?:aße|\.)\s*\d+[a-z]?(?:,?\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)?)',
        r'([A-ZÄÖÜ][a-zäöüß\s]+-?[Ww]eg\s*\d+[a-z]?(?:,?\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)?)',
    ]:
        matches = re.findall(pattern, text)
        if matches:
            valid_addresses = []
            for m in matches[:2]:
                address = m.strip()
                if len(address) > 5 and not is_rolls_royce_address(address):
                    valid_addresses.append(address)
                    print_info(f"Found address: {address}")
            
            if valid_addresses:
                result["SenderAddress"] = ", ".join(valid_addresses)
                break

    return result

def display_tables_beautifully(tables, compact=False):
    """Display tables in a beautiful, readable format"""
    if not tables:
        print_info("No tables to display")
        return
    
    if not compact:
        print_header("📊 EXTRACTED TABLES")
    
    # Check if tables are from AI extraction (structured) or PDF extraction (flat)
    if isinstance(tables, list) and len(tables) > 0:
        # Check if it's AI-extracted format (has headers, rows, title)
        if isinstance(tables[0], dict) and 'headers' in tables[0]:
            # AI-extracted format
            for i, table in enumerate(tables, 1):
                print_header(f"📋 Table {i}: {table.get('title', 'Untitled Table')}")
                
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                
                if headers and rows:
                    # Create a list of dictionaries for tabulate
                    table_data = []
                    for row in rows:
                        row_dict = {}
                        for j, cell in enumerate(row):
                            col_name = headers[j] if j < len(headers) else f"Column_{j+1}"
                            row_dict[col_name] = str(cell) if cell else ""
                        table_data.append(row_dict)
                    
                    # Display with beautiful formatting
                    table_format = "grid" if compact else "fancy_grid"
                    print(tabulate(table_data, headers="keys", tablefmt=table_format, stralign="left"))
                    if not compact:
                        print(f"\n📈 Table {i} Summary: {len(rows)} rows × {len(headers)} columns")
                else:
                    print_warning(f"Table {i} has no data")
                print()
        
        # Check if it's PDF-extracted format (flat dictionaries)
        elif isinstance(tables[0], dict) and 'page' in tables[0]:
            # PDF-extracted format
            print_header("📋 PDF Structure Tables")
            
            # Group by page and table
            grouped_tables = {}
            for table in tables:
                page = table.get('page', 1)
                table_num = table.get('table', 1)
                key = f"Page {page}, Table {table_num}"
                
                if key not in grouped_tables:
                    grouped_tables[key] = []
                grouped_tables[key].append(table)
            
            for table_key, table_rows in grouped_tables.items():
                print_header(f"📋 {table_key}")
                
                # Convert to tabulate format
                table_data = []
                for row in table_rows:
                    row_dict = {k: v for k, v in row.items() if k not in ['page', 'table']}
                    table_data.append(row_dict)
                
                if table_data:
                    table_format = "grid" if compact else "fancy_grid"
                    print(tabulate(table_data, headers="keys", tablefmt=table_format, stralign="left"))
                    if not compact:
                        print(f"\n📈 {table_key} Summary: {len(table_rows)} rows")
                print()
        
        else:
            # Generic format - try to display as is
            if not compact:
                print_header("📋 Generic Tables")
            table_format = "grid" if compact else "fancy_grid"
            print(tabulate(tables, headers="keys", tablefmt=table_format, stralign="left"))
    
    print_success(f"✅ Displayed {len(tables)} table(s)")

def extract_tables_with_pdfplumber(pdf_path):
    if not PDFPLUMBER_AVAILABLE:
        print_warning("pdfplumber not available, skipping tables"); return []
    print_progress("Extracting tables directly from PDF...")
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                for table_num, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        headers, normalized_table = None, []
                        for row in table:
                            cleaned = [str(c).strip().replace('\n',' ') if c else "" for c in row]
                            if headers is None and any(cleaned): headers = cleaned; continue
                            if not any(c.strip() for c in cleaned): continue
                            row_dict = {headers[i] if i < len(headers) else f"Col{i+1}": 
                                       (val[:40]+"..." if len(val)>40 else val) for i,val in enumerate(cleaned)}
                            normalized_table.append(row_dict)
                        if normalized_table:
                            tables.extend(normalized_table)
                            print_info(f"Extracted {len(normalized_table)} rows from page {page_num}, table {table_num+1}")
    except Exception as e:
        print_error(f"Table extraction failed: {e}"); return []
    return tables

def extract_text_from_pdf(pdf_path):
    if not PDFPLUMBER_AVAILABLE:
        print_error("pdfplumber missing, cannot extract text"); return None
    if not os.path.exists(pdf_path):
        print_error(f"PDF not found: {pdf_path}"); return None
    print_progress(f"Extracting text from PDF: {pdf_path}")
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text: text += page_text + "\n"
                print_progress(f"Processed page {i}/{len(pdf.pages)}")
        print_success(f"Extracted {len(text)} characters from {len(pdf.pages)} pages")
    except Exception as e:
        print_error(f"Text extraction failed: {e}"); return None
    return text

def query_local_model(prompt, max_new_tokens=512):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if inputs.shape[1] > 2048:
            inputs = inputs[:, -2048:]; print_warning("Input truncated to 2048 tokens")
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=0.0, top_p=1.0, top_k=0,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in response: response = response.replace(prompt, "").strip()
        return response
    except Exception as e:
        print_error(f"Model query failed: {e}"); return "{}"

# ---------------------------
# Main
# ---------------------------
def process_single_pdf(pdf_path, args, output_dir, thread_lock):
    """Process a single PDF file and return results"""
    try:
        with thread_lock:
            print_progress(f"Processing: {os.path.basename(pdf_path)}")
        
        # Create a temporary copy of args for this thread
        thread_args = argparse.Namespace(**vars(args))
        thread_args.pdf = pdf_path
        
        # Process the PDF
        result = {
            'file': os.path.basename(pdf_path),
            'path': pdf_path,
            'status': 'success',
            'fields': {},
            'tables': [],
            'rolls_royce_detected': False,
            'processing_time': 0,
            'error': None
        }
        
        start_time = datetime.now()
        
        # Extract text
        if not PDFPLUMBER_AVAILABLE:
            result['status'] = 'error'
            result['error'] = 'pdfplumber not available'
            return result
            
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        
        if not text.strip():
            result['status'] = 'error'
            result['error'] = 'No text extracted from PDF'
            return result
        
        # Preprocess text
        processed_text = preprocess_pdf_text(text)
        
        # Check for Rolls-Royce content
        if not thread_args.no_filter:
            detection_result = filter_rolls_royce_content(processed_text)
            result['rolls_royce_detected'] = (
                detection_result.get('contains_rolls_royce_company', False) or
                detection_result.get('contains_rolls_royce_address', False)
            )
        
        # Extract fields
        if not thread_args.table_only:
            if not thread_args.no_filter:
                # Enhanced query with Rolls-Royce filtering
                query = """
Extract the following fields as strict JSON, but EXCLUDE any information about Rolls-Royce Power Systems AG or Rolls-Royce Solutions GmbH:

IMPORTANT: Do NOT extract:
- Company name: "Rolls-Royce Power Systems AG" or "Rolls-Royce Solutions GmbH"
- Address: "Maybachplatz 1, 88045 Friedrichshafen, Germany" or any Rolls-Royce address

Extract only the SENDER/ORIGINATOR company and address (not the recipient):

{ "Date": "", "Angebot": "", "SenderCompany": "", "SenderAddress": "" }
"""
            else:
                query = """
Extract the following fields as strict JSON:
{ "Date": "", "Angebot": "", "SenderCompany": "", "SenderAddress": "" }
"""
            
            fields_response = query_local_model(f"PDF Content:\n{processed_text}\n\n{query}", max_new_tokens=256)
            try:
                parsed_fields = json.loads(fields_response)
                
                # Additional filtering if enabled
                if not thread_args.no_filter:
                    if is_rolls_royce_company(parsed_fields.get('SenderCompany', '')):
                        parsed_fields['SenderCompany'] = ''
                    if is_rolls_royce_address(parsed_fields.get('SenderAddress', '')):
                        parsed_fields['SenderAddress'] = ''
                
                result['fields'] = parsed_fields
            except:
                # Fallback to regex extraction
                result['fields'] = extract_fields_with_regex(processed_text)
        
        # Extract tables
        if not thread_args.fields_only:
            try:
                # Use the table extraction logic from the main function
                table_query = """
Extract all tables as JSON array. Each table should have:
- "headers": array of column headers
- "rows": array of row arrays
- "title": table title/description

Return as: [{"headers": [...], "rows": [...], "title": "..."}]
"""
                tables_response = query_local_model(f"PDF Content:\n{processed_text}\n\n{table_query}", max_new_tokens=1024)
                try:
                    result['tables'] = json.loads(tables_response)
                except:
                    # Fallback to PDF extraction
                    result['tables'] = extract_tables_with_pdfplumber(pdf_path)
            except Exception as e:
                result['tables'] = []
        
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        with thread_lock:
            print_success(f"✅ Completed: {os.path.basename(pdf_path)} ({result['processing_time']:.1f}s)")
        
        return result
        
    except Exception as e:
        result = {
            'file': os.path.basename(pdf_path),
            'path': pdf_path,
            'status': 'error',
            'fields': {},
            'tables': [],
            'rolls_royce_detected': False,
            'processing_time': 0,
            'error': str(e)
        }
        
        with thread_lock:
            print_error(f"❌ Failed: {os.path.basename(pdf_path)} - {str(e)}")
        
        return result

def process_folder_batch(folder_path, args):
    """Process all PDF files in a folder"""
    print_header("BATCH PROCESSING MODE")
    print_info(f"📁 Processing folder: {folder_path}")
    print_info(f"📊 Output directory: {args.output_dir}")
    print_info(f"🔄 Parallel processes: {args.parallel}")
    
    # Find all PDF files
    pdf_patterns = [
        os.path.join(folder_path, "*.pdf"),
        os.path.join(folder_path, "*.PDF"),
        os.path.join(folder_path, "**", "*.pdf"),
        os.path.join(folder_path, "**", "*.PDF")
    ]
    
    pdf_files = []
    for pattern in pdf_patterns:
        pdf_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    pdf_files = sorted(list(set(pdf_files)))
    
    if not pdf_files:
        print_error("❌ No PDF files found in the specified folder")
        return
    
    print_info(f"📄 Found {len(pdf_files)} PDF files to process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create summary file
    summary_file = os.path.join(args.output_dir, "batch_processing_summary.json")
    
    # Process files
    results = []
    thread_lock = threading.Lock()
    
    if args.parallel > 1:
        print_info(f"🚀 Processing with {args.parallel} parallel threads...")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf_path, args, args.output_dir, thread_lock): pdf_path
                for pdf_path in pdf_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_pdf):
                result = future.result()
                results.append(result)
    else:
        print_info("🔄 Processing sequentially...")
        for pdf_path in pdf_files:
            result = process_single_pdf(pdf_path, args, args.output_dir, thread_lock)
            results.append(result)
    
    # Generate summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    rolls_royce_detected = [r for r in results if r.get('rolls_royce_detected', False)]
    
    print_header("BATCH PROCESSING SUMMARY")
    print_info(f"📊 Total files processed: {len(results)}")
    print_success(f"✅ Successful: {len(successful)}")
    print_error(f"❌ Failed: {len(failed)}")
    print_warning(f"🚫 Rolls-Royce content detected: {len(rolls_royce_detected)}")
    
    if failed:
        print_header("FAILED FILES")
        for result in failed:
            print_error(f"  ❌ {result['file']}: {result['error']}")
    
    if rolls_royce_detected:
        print_header("ROLLS-ROYCE CONTENT DETECTED")
        for result in rolls_royce_detected:
            print_warning(f"  🚫 {result['file']}")
    
    # Save individual results and display tables
    for result in results:
        if result['status'] == 'success':
            # Save fields
            if result['fields']:
                fields_file = os.path.join(args.output_dir, f"{os.path.splitext(result['file'])[0]}_fields.json")
                with open(fields_file, 'w', encoding='utf-8') as f:
                    json.dump(result['fields'], f, indent=2, ensure_ascii=False)
            
            # Save tables and display them
            if result['tables']:
                tables_file = os.path.join(args.output_dir, f"{os.path.splitext(result['file'])[0]}_tables.json")
                with open(tables_file, 'w', encoding='utf-8') as f:
                    json.dump(result['tables'], f, indent=2, ensure_ascii=False)
                
                # Display tables for this file
                print_header(f"📊 TABLES FROM: {result['file']}")
                display_tables_beautifully(result['tables'], compact=args.compact_tables)
    
    # Save summary
    summary = {
        'processing_date': datetime.now().isoformat(),
        'folder_processed': folder_path,
        'total_files': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'rolls_royce_detected': len(rolls_royce_detected),
        'results': results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print_success(f"📁 Results saved to: {args.output_dir}")
    print_success(f"📊 Summary saved to: {summary_file}")

def test_rolls_royce_filtering():
    """Test function to demonstrate Rolls-Royce filtering functionality"""
    print_header("TESTING ROLLS-ROYCE FILTERING")
    
    # Test company filtering
    test_companies = [
        "Rolls-Royce Power Systems AG",
        "Rolls-Royce Solutions GmbH", 
        "BMW AG",
        "Siemens AG",
        "Rolls Royce Power Systems AG",  # With space instead of hyphen
        "ROLLS-ROYCE POWER SYSTEMS AG"   # All caps
    ]
    
    print_info("Testing company name filtering:")
    for company in test_companies:
        is_filtered = is_rolls_royce_company(company)
        status = "🚫 FILTERED" if is_filtered else "✅ ALLOWED"
        print(f"  {status}: {company}")
    
    # Test address filtering
    test_addresses = [
        "Maybachplatz 1, 88045 Friedrichshafen, Germany",
        "Musterstraße 123, 12345 Berlin, Germany",
        "Rolls-Royce Power Systems AG, Maybachplatz 1, 88045 Friedrichshafen",
        "Hauptstraße 45, 54321 München, Germany"
    ]
    
    print_info("\nTesting address filtering:")
    for address in test_addresses:
        is_filtered = is_rolls_royce_address(address)
        status = "🚫 FILTERED" if is_filtered else "✅ ALLOWED"
        print(f"  {status}: {address}")
    
    print_success("Rolls-Royce filtering test completed!")

def main():
    global tokenizer, model
    args = setup_args()
    if args.verbose: logging.basicConfig(level=logging.INFO)
    
    # Handle convenience parallel processing options
    if args.fast:
        args.parallel = 4
        print_info("🚀 Fast mode enabled: Using 4 parallel processes")
    elif args.slow:
        args.parallel = 1
        print_info("🐌 Slow mode enabled: Using 1 parallel process")
    elif args.turbo:
        args.parallel = 6
        print_info("⚡ Turbo mode enabled: Using 6 parallel processes")

    print_header("DEEPSEEK PDF PROCESSING SYSTEM")
    print_info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"🔄 Parallel processes: {args.parallel}")
    
    # Display processing mode
    if args.parallel == 1:
        print_info("🐌 Processing mode: SLOW (1 process)")
    elif args.parallel == 3:
        print_info("⚙️  Processing mode: STANDARD (3 processes)")
    elif args.parallel == 4:
        print_info("🚀 Processing mode: FAST (4 processes)")
    elif args.parallel >= 6:
        print_info("⚡ Processing mode: TURBO (6+ processes)")
    else:
        print_info(f"🔧 Processing mode: CUSTOM ({args.parallel} processes)")
    
    # Show filtering status
    if not args.no_filter:
        print_info("🛡️  Rolls-Royce filtering ENABLED")
    else:
        print_info("🔓 Rolls-Royce filtering DISABLED")
    
    # Run filter test if requested
    if args.test_filter:
        test_rolls_royce_filtering()
        return
    
    # Handle batch processing if folder is specified or if default path is a folder
    folder_to_process = args.folder or (os.path.isdir(args.pdf) and args.pdf)
    
    if folder_to_process:
        if not os.path.exists(folder_to_process):
            print_error(f"❌ Folder does not exist: {folder_to_process}")
            return
        
        if not os.path.isdir(folder_to_process):
            print_error(f"❌ Path is not a directory: {folder_to_process}")
            return
        
        # Load model for batch processing
        print_progress("Loading model for batch processing...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            # Try with device_map first, fallback to CPU if accelerate not available
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, 
                    dtype=torch.float16, 
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as device_error:
                print_warning(f"⚠️  Device mapping failed, using CPU: {device_error}")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, 
                    dtype=torch.float16, 
                    trust_remote_code=True
                )
            print_success("✅ Model loaded successfully")
        except Exception as e:
            print_error(f"❌ Failed to load model: {e}")
            return
        
        # Process folder
        process_folder_batch(folder_to_process, args)
        return

    # Load model
    print_header("MODEL INITIALIZATION")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float16)
        print_success("DeepSeek model loaded successfully (offline mode)")
    except Exception as e:
        print_error(f"Error loading model from {args.model}: {e}"); sys.exit(1)

    # Extract text
    print_header("PDF PROCESSING")
    pdf_text = extract_text_from_pdf(args.pdf)
    if not pdf_text: sys.exit(1)
    processed_text = preprocess_pdf_text(pdf_text)

    extracted_fields, extracted_tables = {}, []

    # Fields
    if not args.table_only:
        print_header("FIELD EXTRACTION")
        
        # First, use DeepSeek to intelligently filter out Rolls-Royce content (unless disabled)
        detection_result = {"contains_rolls_royce_company": False, "contains_rolls_royce_address": False, "confidence": 0.0}
        
        if not args.no_filter:
            print_progress("Using DeepSeek to detect and filter Rolls-Royce content...")
            detection_result = filter_rolls_royce_content(processed_text)
            
            if detection_result.get("contains_rolls_royce_company", False):
                print_warning("⚠️  Rolls-Royce company detected - will be filtered out")
            if detection_result.get("contains_rolls_royce_address", False):
                print_warning("⚠️  Rolls-Royce address detected - will be filtered out")
        else:
            print_info("🔓 Rolls-Royce filtering disabled (--no-filter flag used)")
        
        # Enhanced query with conditional Rolls-Royce filtering instructions
        if not args.no_filter:
            query = """
Extract the following fields as strict JSON, but EXCLUDE any information about Rolls-Royce Power Systems AG or Rolls-Royce Solutions GmbH:

IMPORTANT: Do NOT extract:
- Company name: "Rolls-Royce Power Systems AG" or "Rolls-Royce Solutions GmbH"
- Address: "Maybachplatz 1, 88045 Friedrichshafen, Germany" or any Rolls-Royce address

Extract only the SENDER/ORIGINATOR company and address (not the recipient):

{ "Date": "", "Angebot": "", "SenderCompany": "", "SenderAddress": "" }
"""
        else:
            query = """
Extract the following fields as strict JSON:

{ "Date": "", "Angebot": "", "SenderCompany": "", "SenderAddress": "" }
"""
        
        fields_response = query_local_model(f"PDF Content:\n{processed_text}\n\n{query}", max_new_tokens=256)
        try:
            parsed_fields = json.loads(fields_response)
            
            # Additional filtering to ensure no Rolls-Royce content slipped through (only if filtering is enabled)
            if not args.no_filter:
                if is_rolls_royce_company(parsed_fields.get("SenderCompany", "")):
                    print_warning("Filtered out Rolls-Royce company from DeepSeek result")
                    parsed_fields["SenderCompany"] = ""
                
                if is_rolls_royce_address(parsed_fields.get("SenderAddress", "")):
                    print_warning("Filtered out Rolls-Royce address from DeepSeek result")
                    parsed_fields["SenderAddress"] = ""
                
        except:
            print_warning("DeepSeek field extraction failed, using regex fallback...")
            if args.no_filter:
                # Use original regex without filtering
                parsed_fields = extract_fields_with_regex(processed_text)
            else:
                # Use regex with filtering
                parsed_fields = extract_fields_with_regex(processed_text)
        
        extracted_fields = parsed_fields
        
        # Show filtering summary
        if not args.no_filter:
            print_header("ROLLS-ROYCE FILTERING SUMMARY")
            print(f"🔍 Detection confidence: {detection_result.get('confidence', 0.0):.1%}")
            print(f"🏢 Rolls-Royce company detected: {'Yes' if detection_result.get('contains_rolls_royce_company', False) else 'No'}")
            print(f"📍 Rolls-Royce address detected: {'Yes' if detection_result.get('contains_rolls_royce_address', False) else 'No'}")
            
            # Show what was filtered out
            if detection_result.get('contains_rolls_royce_company', False) or detection_result.get('contains_rolls_royce_address', False):
                print_info("✅ Rolls-Royce content successfully filtered out from extraction results")
        else:
            print_header("EXTRACTION SUMMARY (NO FILTERING)")
            print_info("🔓 All content extracted without filtering")
        
        print_header("EXTRACTED FIELDS")
        print(f"📅 Date: {parsed_fields.get('Date','')}")
        print(f"📄 Quotation: {parsed_fields.get('Angebot','')}")
        print(f"🏢 Company: {parsed_fields.get('SenderCompany','')}")
        print(f"📍 Address: {parsed_fields.get('SenderAddress','')}")

    # Tables
    if not args.fields_only:
        print_header("TABLE EXTRACTION")
        extracted_tables = extract_tables_with_pdfplumber(args.pdf)
        if extracted_tables:
            display_tables_beautifully(extracted_tables, compact=args.compact_tables)

    # Save JSON if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"fields": extracted_fields, "tables": extracted_tables}, f, indent=2, ensure_ascii=False)
        print_success(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()


