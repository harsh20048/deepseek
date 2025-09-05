import pdfplumber
from embedchain import App
import json
from tabulate import tabulate   # <-- for pretty tables
import os
from datetime import datetime
import sys

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
# 2. Configure Embedchain with DeepSeek (Local Offline)
# ---------------------------
print_header("DEEPSEEK PDF PROCESSING SYSTEM")
print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Get the current directory where the model files are located
model_path = os.path.dirname(os.path.abspath(__file__))

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
pdf_text = extract_text_from_pdf("german_quotation.pdf")

if pdf_text is None:
    print_error("Failed to extract text from PDF. Exiting...")
    sys.exit(1)

print_success("PDF content extracted successfully")

def query_local_model(prompt, max_length=512):
    """Query the local DeepSeek model"""
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print_error(f"Error querying model: {e}")
        return "Error processing query"

# ---------------------------
# 4. Query for structured fields
# ---------------------------
print_header("FIELD EXTRACTION")
print_progress("Extracting structured fields from German quotation...")

query = """
Extract the following fields from this German quotation PDF:
- Date (Datum)
- Angebot (quotation number)
- Sender company name (Firma)
- Sender address (Straße, PLZ, Ort)

Return result in JSON:
{
  "Date": "",
  "Angebot": "",
  "SenderCompany": "",
  "SenderAddress": ""
}
"""

print_info("Querying DeepSeek model for field extraction...")
full_prompt = f"PDF Content:\n{pdf_text}\n\n{query}"
fields_response = query_local_model(full_prompt)

print_header("EXTRACTED FIELDS")
print(f"{Colors.OKCYAN}{fields_response}{Colors.ENDC}")

# ---------------------------
# 5. Query for tables (with 40-char rule)
# ---------------------------
print_header("TABLE EXTRACTION")
print_progress("Extracting tabular data from German quotation...")

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
full_table_prompt = f"PDF Content:\n{pdf_text}\n\n{table_query}"
table_response = query_local_model(full_table_prompt)

# ---------------------------
# 6. Truncate in Python (extra safety)
# ---------------------------
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

print_progress("Processing and cleaning table data...")
cleaned_table = truncate_table_json(table_response, max_len=40)

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
print_header("PROCESSING COMPLETE")
print_success(f"PDF processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_info("All data processed locally with DeepSeek model")
print_success("🔒 Data privacy maintained - no external data transmission")
print(f"\n{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
print(f"{Colors.OKGREEN}{'PDF PROCESSING SUCCESSFUL!'.center(80)}{Colors.ENDC}")
print(f"{Colors.OKGREEN}{'='*80}{Colors.ENDC}")