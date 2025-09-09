import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import difflib
import re
from typing import Dict, List, Tuple, Optional

# Try to import advanced dependencies, fallback to test.py if they fail
try:
    from transformers import (
        LayoutLMv3Processor, 
        LayoutLMv3ForTokenClassification, 
        TrainingArguments, 
        Trainer,
        EvalPrediction,
        AutoTokenizer, 
        AutoModelForCausalLM
    )
    from datasets import Dataset
    from tqdm import tqdm
    import torch
    from pdfminer.high_level import extract_pages, extract_text
    from pdfminer.layout import LTTextBoxHorizontal, LTChar, LTTextLine
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    from pdf2image import convert_from_path
    import spacy
    ADVANCED_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced dependencies not available: {e}")
    print("🔄 Falling back to test.py extraction methods...")
    ADVANCED_DEPS_AVAILABLE = False

# Suppress warnings
try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
except:
    pass

# ============================================================================
# FALLBACK EXTRACTION METHODS FROM TEST.PY
# ============================================================================

def extract_fields_with_regex_fallback(text):
    """Enhanced fallback field extraction using improved patterns from test.py"""
    print("🔄 Using enhanced regex fallback extractor from test.py...")
    result = {"date": "", "angebot": "", "company_name": "", "sender_address": ""}
    text = re.sub(r'\s+', ' ', text.strip())

    # Enhanced Date extraction with more patterns
    for pattern in [
        r'Datum[:\s]*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'Date[:\s]*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'vom[:\s]*(\d{1,2}[./]\d{1,2}[./]\d{4})',
        r'(\d{1,2}[./]\d{1,2}[./]\d{4})',  # General date pattern last
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["date"] = match.group(1)
            break

    # Enhanced Angebot extraction with more specific patterns
    for pattern in [
        r'Angebots?-?Nr\.?[:\s]*([A-Z]{2,3}-?\d{4,})',  # Pattern like AN-22449
        r'Angebots?-?Nr\.?[:\s]*(\d{6,})',  # Pattern like 202200808
        r'Angebot(?:s?-?Nr\.?|s?nummer)?[:\s]*([A-Za-z0-9.\-/]{4,})',
        r'Offerte[:\s]*([A-Za-z0-9.\-/]{4,})',
        r'Quotation[:\s]*([A-Za-z0-9.\-/]{4,})',
        r'Nr\.?\s*([A-Z]{2,3}-?\d{4,})',  # Pattern like AN-22449
        r'([A-Z]{2,3}-?\d{4,})',  # Direct pattern like AN-22449
        r'(\d{6,})',  # Long numbers like 202200808
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and len(match.group(1)) >= 4:
            angebot = match.group(1).strip()
            # Filter out common false positives
            if angebot not in ["Nr.", "unter", "Datum", "Sehr", "Seite"]:
                result["angebot"] = angebot
                break

    # Enhanced Company extraction with better filtering
    company_patterns = [
        r'([A-ZÄÖÜ][A-Za-zäöüß\s&\.]+(?:GmbH|AG|UG|KG|OHG|mbH)(?:\s*&\s*Co\.?\s*KG)?)',
        r'([A-ZÄÖÜ][A-Za-zäöüß\s&\.]+(?:GmbH|AG|UG|KG|OHG|mbH))',
        r'([A-Z][^.\n]*(?:GmbH|AG|UG|KG|OHG|mbH))',
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for company in matches:
                company = company.strip()
                # Better filtering - exclude common false positives
                if (len(company) > 8 and 
                    not company.startswith("Seite") and 
                    not company.startswith("von") and
                    "Sparkasse" not in company and
                    "Konto" not in company):
                    result["company_name"] = company
                    break
            if result["company_name"]:
                break

    # Enhanced Address extraction with better patterns
    address_patterns = [
        # Street patterns
        r'([A-ZÄÖÜ][a-zäöüß\s]+-?(?:[Ss]tr(?:aße|\.)|[Ww]eg|[Pp]latz)\s*\d+[a-z]?(?:,?\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)?)',
        # Postal code + city
        r'(\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)',
        # Combined address patterns
        r'([A-ZÄÖÜ][a-zäöüß\s]+\d+[a-z]?\s*,?\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß\s]+)',
    ]
    
    for pattern in address_patterns:
        matches = re.findall(pattern, text)
        if matches:
            valid_addresses = []
            for address in matches[:3]:  # Check first 3 matches
                address = address.strip()
                # Better filtering for addresses
                if (len(address) > 10 and 
                    not address.startswith("Seite") and
                    not "Angebot" in address and
                    not "Sparkasse" in address):
                    valid_addresses.append(address)
            
            if valid_addresses:
                result["sender_address"] = "\n".join(valid_addresses[:2])  # Take top 2
                break

    return result

def extract_fields_with_deepseek_fallback(text, model_path=None):
    """Fallback field extraction using DeepSeek from test.py"""
    if not ADVANCED_DEPS_AVAILABLE:
        print("🔄 DeepSeek not available, using regex fallback...")
        return extract_fields_with_regex_fallback(text)
    
    try:
        print("🤖 Using DeepSeek fallback extractor from test.py...")
        
        # Try to load DeepSeek model
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "deepseek_model"
        
        if not Path(model_path).exists():
            print(f"⚠️ DeepSeek model not found at {model_path}, using regex fallback...")
            return extract_fields_with_regex_fallback(text)
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, dtype=torch.float16)
        
        # DeepSeek extraction query
        query = """
Extract the following fields as strict JSON:
Extract only the SENDER/ORIGINATOR company and address (not the recipient):

{ "date": "", "angebot": "", "company_name": "", "sender_address": "" }

Text to analyze:
""" + text[:2000]  # Limit text length
        
        # Generate response
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed_fields = json.loads(json_str)
                
                # Convert to expected format
                result = {
                    "date": parsed_fields.get("date", ""),
                    "angebot": parsed_fields.get("angebot", ""),
                    "company_name": parsed_fields.get("company_name", ""),
                    "sender_address": parsed_fields.get("sender_address", "")
                }
                return result
        except:
            pass
        
        # If DeepSeek fails, use regex fallback
        print("⚠️ DeepSeek parsing failed, using regex fallback...")
        return extract_fields_with_regex_fallback(text)
        
    except Exception as e:
        print(f"⚠️ DeepSeek extraction failed: {e}, using regex fallback...")
        return extract_fields_with_regex_fallback(text)

# =========================
# CONFIGURATION
# =========================
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Use relative paths from project root
project_root = Path(__file__).parent.parent
local_model_path = project_root / "models" / "final_model"
pdf_folder = project_root / "pdfs"
tables_folder = project_root / "tables"
labels_excel = project_root / "labels.xlsx"
output_dir = project_root / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# Define label mappings
LABEL_LIST = [
    "O", "B-DATE", "I-DATE", "B-COMPANY", "I-COMPANY", 
    "B-ADDRESS", "I-ADDRESS", "B-ANGEBOT", "I-ANGEBOT", 
    "B-TABLE", "I-TABLE"
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# =========================
# UTILITY FUNCTIONS FOR SAFE VALUE HANDLING
# =========================

def safe_str(value) -> str:
    """Safely convert value to string, handling lists and None"""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v) if value else ""
    elif value is None:
        return ""
    else:
        return str(value)

def print_extraction_results(results: Dict, title: str = "EXTRACTION RESULTS"):
    """Safely print extraction results, handling list values"""
    print(f"\nâœ… {title}:")
    print("=" * 50)
    for field, value in results.items():
        if field != "Table_File" and not field.endswith("_Source"):
            value_str = safe_str(value)
            source = results.get(f"{field}_Source", "")
            status = "âœ“" if value_str and value_str.strip() else "âœ—"
            source_display = f" [{source}]" if source else ""
            print(f"{status} {field:15}: {value_str}{source_display}")
    print("=" * 50)

# =========================
# ENHANCED RULE-BASED EXTRACTOR WITH POSITIONAL ANGEBOT EXTRACTION
# =========================

# Import OCR letterhead function with error handling
try:
    from extraction.text_extraction import extract_letterhead_text
except ImportError:
    extract_letterhead_text = None

# Load spaCy models (German and English)
try:
    nlp_de = spacy.load("de_core_news_sm")
except:
    nlp_de = None

try:
    nlp_en = spacy.load("en_core_web_sm")
except:
    nlp_en = None

class PDFFieldExtractor:
    def __init__(self, receiver_company: str = "Rolls Royce Solutions Gmbh"):
        self.receiver_company = receiver_company.lower()

        # Keywords for angebot (quote) number - added more variations
        self.angebot_keywords = [
            "angebot", "angebotsnummer", "angebot.nr", "angebot-nr", "angebot_nr",
            "referenz", "referenznummer", "ref", "ref-nr", "ref.nr",
            "offer.no", "offer.number", "offer", "quotation.no", "quotation",
            "quote.no", "quote", "nr", "no", "number", "proposal", "business"
        ]

        # Date patterns
        self.date_patterns = [
            r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
            r"\b\d{1,2}\.\s*\w+\s*\d{4}\b"
        ]

        # Enhanced postal code patterns for various countries
        self.postal_patterns = [
            r'\b\d{4,5}\s+[A-Za-zÃ¤Ã¶Ã¼Ã„Ã–ÃœÃŸ\s\-]{2,30}\b',  # German
            r'\b[A-Za-z]{1,2}\d{1,2}\s*\d[A-Za-z]{2}\b',  # UK
            r'\b\d{5}(-\d{4})?\b',                          # US
            r'\b\d{4}\s*[A-Z]{2}\b',                        # Netherlands
            r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b',              # Canada
        ]

        self.company_indicators = [
            'gmbh', 'ag', 'kg', 'inc', 'corp', 'ltd', 'llc', 'co', 'company',
            'gmbh & co', 'gmbh & co.', 'co. kg', 'co.kg', 'e.v.', 'ev',
            'ohg', 'kg', 'partg', 'ug', 'limited', 'corporation', 'incorporated'
        ]

        # Address-related keywords in multiple languages
        self.address_keywords = {
            'street': ['straÃŸe', 'str.', 'str', 'strasse', 'street', 'road', 'avenue',
                       'ave', 'boulevard', 'blvd', 'lane', 'ln', 'drive', 'dr',
                       'way', 'weg', 'platz', 'place', 'pl', 'allee', 'gasse'],
            'contact': ['tel', 'telefon', 'phone', 'fax', 'email', 'mail', 'web',
                        'www', 'mobile', 'mobil', 'handy'],
            'identifiers': ['ustid', 'ust-id', 'vat', 'steuer', 'tax', 'register',
                            'hrb', 'hra', 'amtsgericht', 'court', 'reg', 'nr'],
            'country': ['deutschland', 'germany', 'austria', 'Ã¶sterreich', 'schweiz',
                        'switzerland', 'uk', 'united kingdom', 'usa', 'france', 'india']
        }

    def extract_date(self, text: str) -> str:
        """Extract date from text using various date patterns"""
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        return ""

    def extract_angebot(self, text: str) -> str:
        """ENHANCED Extract angebot number - handles values positioned to the right of keywords"""
        print(f"DEBUG: extract_angebot called")
        print(f"DEBUG: Text preview (first 1000 chars): {text[:1000]}")

        # Method 1: Positional extraction for form/table layouts (NEW - handles your case)
        result = self._extract_angebot_positional(text)
        if result:
            print(f"DEBUG: Found angebot via positional method: '{result}'")
            return result

        # Method 2: Your original inline patterns
        result = self._extract_angebot_inline(text)
        if result:
            print(f"DEBUG: Found angebot via inline method: '{result}'")
            return result

        # Method 3: Fallback patterns
        result = self._extract_angebot_fallback(text)
        if result:
            print(f"DEBUG: Found angebot via fallback method: '{result}'")
            return result

        print("DEBUG: No angebot found")
        return ""

    def _extract_angebot_positional(self, text: str) -> str:
        """NEW METHOD: Extract angebot when value is positioned to the right of keyword"""
        print("DEBUG: Trying positional angebot extraction...")
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            print(f"DEBUG: Checking line {i}: '{line_clean}'")
            
            # Check if line contains angebot keyword
            for keyword in self.angebot_keywords:
                if keyword.lower() in line_clean.lower():
                    print(f"DEBUG: Found keyword '{keyword}' in line: '{line_clean}'")
                    
                    # Strategy 1: Value in same line after keyword (e.g., "Angebot Nr. 123456")
                    same_line_match = self._extract_from_same_line(line_clean, keyword)
                    if same_line_match:
                        return same_line_match
                    
                    # Strategy 2: Value in adjacent lines (common in forms)
                    adjacent_match = self._extract_from_adjacent_lines(lines, i, keyword)
                    if adjacent_match:
                        return adjacent_match
                    
                    # Strategy 3: Value in tabular format (split by tabs/spaces)
                    tabular_match = self._extract_from_tabular(line_clean, keyword)
                    if tabular_match:
                        return tabular_match
        
        return ""

    def _extract_from_same_line(self, line: str, keyword: str) -> str:
        """Extract value from same line as keyword"""
        print(f"DEBUG: Extracting from same line with keyword '{keyword}': '{line}'")
        
        # Pattern 1: "Angebot Nr. 123456" or "Angebot Nr.: 123456"
        patterns = [
            rf"{keyword}[\s\.]*nr[\s\.:]*([A-Z0-9\-/\.#_]{{3,20}})",
            rf"{keyword}[\s\.:]+([A-Z0-9\-/\.#_]{{3,20}})",
            rf"{keyword}[\s]*[:\.-]\s*([A-Z0-9\-/\.#_]{{3,20}})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                print(f"DEBUG: Same-line candidate: '{candidate}'")
                if self._valid_angebot(candidate):
                    print(f"DEBUG: Valid same-line angebot: '{candidate}'")
                    return candidate
        
        return ""

    def _extract_from_adjacent_lines(self, lines: List[str], keyword_line_idx: int, keyword: str) -> str:
        """Extract value from lines adjacent to keyword line (common in forms)"""
        print(f"DEBUG: Extracting from adjacent lines around line {keyword_line_idx}")
        
        # Check next 3 lines for potential values
        for offset in range(1, 4):
            if keyword_line_idx + offset < len(lines):
                next_line = lines[keyword_line_idx + offset].strip()
                if next_line:
                    print(f"DEBUG: Checking adjacent line +{offset}: '{next_line}'")
                    
                    # Look for standalone alphanumeric values
                    candidates = re.findall(r'\b([A-Z0-9\-/\.#_]{3,20})\b', next_line.upper())
                    for candidate in candidates:
                        print(f"DEBUG: Adjacent line candidate: '{candidate}'")
                        if self._valid_angebot(candidate):
                            print(f"DEBUG: Valid adjacent angebot: '{candidate}'")
                            return candidate
        
        return ""

    def _extract_from_tabular(self, line: str, keyword: str) -> str:
        """Extract value when line is in tabular format (separated by tabs/multiple spaces)"""
        print(f"DEBUG: Extracting from tabular format: '{line}'")
        
        # Split by multiple spaces or tabs (common in forms)
        parts = re.split(r'\s{2,}|\t+', line)
        if len(parts) >= 2:
            print(f"DEBUG: Tabular parts: {parts}")
            
            # Find part with keyword, then look for value in other parts
            keyword_found = False
            for part in parts:
                if keyword.lower() in part.lower():
                    keyword_found = True
                    continue
                
                if keyword_found:
                    # Look for values in subsequent parts
                    candidates = re.findall(r'\b([A-Z0-9\-/\.#_]{3,20})\b', part.upper())
                    for candidate in candidates:
                        print(f"DEBUG: Tabular candidate: '{candidate}'")
                        if self._valid_angebot(candidate):
                            print(f"DEBUG: Valid tabular angebot: '{candidate}'")
                            return candidate
        
        return ""

    def _extract_angebot_inline(self, text: str) -> str:
        """Your original inline pattern extraction"""
        print("DEBUG: Trying inline angebot extraction...")
        
        # Pre-process text to handle special cases
        text_processed = text.replace('\n', ' ').replace('\r', ' ')
        text_processed = re.sub(r'\s+', ' ', text_processed)

        patterns = []
        for keyword in self.angebot_keywords:
            patterns.extend([
                # Standard patterns: keyword followed by number
                rf"{keyword}[^\w]*[:\s#-]\s*([A-Z0-9\-/\.#_]{{3,20}})",
                rf"{keyword}\s+([A-Z0-9\-/\.#_]{{3,20}})",
                rf"{keyword}[:\s#-]*([A-Z0-9\-/\.#_]{{3,20}})",

                # Reverse patterns: number followed by keyword
                rf"([A-Z0-9]{{6,20}})\s*[-\s]*\w*\s*[-\s]*{keyword}\b",
                rf"([A-Z0-9]{{6,20}})\s*[-\s]+.*?{keyword}\b",
            ])

        for pattern in patterns:
            matches = re.findall(pattern, text_processed.upper(), re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if self._valid_angebot(match):
                    return match

        return ""

    def _extract_angebot_fallback(self, text: str) -> str:
        """Fallback extraction methods"""
        print("DEBUG: Trying fallback extraction...")
        
        lines = text.split('\n')
        for line in lines:
            line_upper = line.upper()
            for keyword in self.angebot_keywords:
                if keyword.upper() in line_upper:
                    print(f"DEBUG: Found keyword '{keyword}' in line: '{line}'")
                    # Extract all potential numbers from this line
                    numbers = re.findall(r'\b([A-Z0-9\-/\.#_]{3,20})\b', line_upper)
                    print(f"DEBUG: Numbers found in line: {numbers}")
                    for num in numbers:
                        if self._valid_angebot(num):
                            print(f"DEBUG: Valid angebot from fallback: '{num}'")
                            return num

        return ""

    def _valid_angebot(self, candidate: str) -> bool:
        """Enhanced validation with date rejection"""
        candidate = candidate.strip()
        print(f"DEBUG: Validating angebot candidate: '{candidate}'")

        # First check if it looks like a date (to avoid extracting dates as angebot)
        date_patterns = [
            r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$',  # DD/MM/YYYY, DD-MM-YYYY
            r'^\d{4}-\d{1,2}-\d{1,2}$',            # YYYY-MM-DD
            r'^\d{1,2}\.\d{1,2}\.\d{4}$',          # DD.MM.YYYY
        ]
        
        for date_pattern in date_patterns:
            if re.match(date_pattern, candidate):
                print(f"DEBUG: Rejected '{candidate}' - looks like a date")
                return False

        # Standard angebot validation
        is_valid = (
            3 <= len(candidate) <= 20 and
            any(c.isdigit() for c in candidate) and
            all(c.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/.#_' for c in candidate) and
            not candidate.isalpha() and
            sum(c.isdigit() for c in candidate) >= 2  # At least 2 digits
        )

        print(f"DEBUG: Validation result for '{candidate}': {is_valid}")
        return is_valid

    def _truncate_at_company_indicator(self, text: str) -> str:
        """Truncate text at company indicator and return only the part before it."""
        sorted_indicators = sorted(self.company_indicators, key=len, reverse=True)

        for indicator in sorted_indicators:
            pattern = rf'\b{re.escape(indicator)}\b'
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                end_pos = match.end()
                truncated = text[:end_pos].strip()
                return truncated

        return text

    def _extract_company_with_indicators(self, text: str) -> str:
        """Enhanced company extraction with better differentiation from your code"""
        if not text:
            return ""

        lines = [line.strip() for line in text.replace('\r', '').split('\n') if line.strip()]

        # Check each line for company indicators
        for i, line in enumerate(lines[:15]):
            if not line:
                continue

            for indicator in self.company_indicators:
                if indicator.lower() in line.lower():
                    truncated_line = self._truncate_at_company_indicator(line)
                    cleaned = re.sub(r'[^\w\s&.,-]', '', truncated_line)
                    cleaned = ' '.join(cleaned.split())

                    if len(cleaned) < 3:
                        continue

                    if self._is_address_part(cleaned) and not any(ind in cleaned.lower() for ind in self.company_indicators):
                        continue

                    if self.receiver_company and self.receiver_company in cleaned.lower():
                        continue

                    print(f"DEBUG: Found company via indicator: '{cleaned}'")
                    return cleaned

        # Fallback to first meaningful line
        for line in lines[:10]:
            if (line and len(line) > 2 and not self._is_address_part(line) and
                not re.match(r'^\d+[./-]\d+[./-]\d+', line) and
                not re.match(r'^(datum|date|von|from|an|to):', line.lower()) and
                (self.receiver_company == "" or self.receiver_company not in line.lower())):
                
                cleaned = re.sub(r'[^\w\s&.,-]', '', line)
                cleaned = ' '.join(cleaned.split())
                return cleaned

        return ""

    def extract_company_name(self, text: str, language: str = "en") -> str:
        """Extract sender company name using your enhanced logic"""
        print(f"DEBUG: extract_company_name called with language: {language}")

        # Step 1: OCR from letterhead image (if available)
        try:
            if extract_letterhead_text is not None and os.path.exists("temp.pdf"):
                letterhead_text = extract_letterhead_text("temp.pdf")
                if letterhead_text:
                    company_name = self._extract_company_with_indicators(letterhead_text)
                    if company_name:
                        return company_name
        except Exception:
            pass

        # Step 2: NLP fallback
        if language == "de" and nlp_de:
            result = self._extract_with_nlp(text, nlp_de)
            if result:
                return result
        elif language == "en" and nlp_en:
            result = self._extract_with_nlp(text, nlp_en)
            if result:
                return result

        # Step 3: Heuristic extraction using your logic
        return self._extract_company_with_indicators(text)

    def _extract_with_nlp(self, text: str, nlp_model) -> str:
        """Extract company name using spaCy NLP model"""
        try:
            doc = nlp_model(text[:1000])
            for ent in doc.ents:
                if (ent.label_ in ["ORG", "PERSON"] and
                    len(ent.text.strip()) > 2 and
                    ent.text.lower() != self.receiver_company):
                    return self._truncate_at_company_indicator(ent.text.strip())
        except Exception as e:
            print(f"NLP extraction failed: {e}")
        return ""

    def _is_address_part(self, text: str) -> bool:
        """Check if text looks like part of an address"""
        patterns = [
            r'\d{4,5}\s+\w+',  # Postal code + city
            r'\b(straÃŸe|str\.|street|rd\.|road|ave\.|avenue|tel\.|phone|fax|email|www\.)\b',
            r'\b\d+\s+(straÃŸe|str|street|rd|road|ave|avenue)\b',
            r'\b(telefon|tel|fax|email|web|www)\s*[:]\s*',
            r'^\d+\s*[a-zA-Z]?\s*$',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def extract_sender_address(self, text: str, language: str = "en") -> str:
        """Enhanced sender address extraction with better filtering"""
        print("DEBUG: Starting enhanced sender address extraction")

        # Strategy 1: Extract using multiple methods with better filtering
        address_candidates = []

        # Method 1: Extract from top-left area (most reliable for letterheads)
        top_left_address = self._extract_from_top_left(text)
        if top_left_address and len(top_left_address) < 200:  # Reasonable length filter
            address_candidates.append(('top_left', top_left_address, 0.8))

        # Method 2: Extract from top lines (good for standard layouts)
        top_lines_address = self._extract_from_top_lines(text)
        if top_lines_address and len(top_lines_address) < 200:
            address_candidates.append(('top_lines', top_lines_address, 0.6))

        # Method 3: Extract using patterns (fallback)
        pattern_address = self._extract_using_patterns(text)
        if pattern_address and len(pattern_address) < 200:
            address_candidates.append(('patterns', pattern_address, 0.4))

        print(f"DEBUG: Found {len(address_candidates)} address candidates")
        for i, (method, addr, score) in enumerate(address_candidates):
            print(f"DEBUG: Candidate {i+1} ({method}): '{addr[:50]}...' (score: {score})")

        # Select best candidate
        if address_candidates:
            best_address = self._select_best_address(address_candidates)
            print(f"DEBUG: Selected best address: {best_address[:100]}...")
            return best_address

        # Fallback to simple extraction
        fallback = self._simple_address_extraction(text)
        print(f"DEBUG: Fallback address: {fallback}")
        return fallback

    def _extract_from_top_left(self, text: str) -> Optional[str]:
        """Extract address from top-left area of document with length limits"""
        lines = text.split('\n')
        address_lines = []
        
        # Only look at first 15 lines for sender address
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if not line:
                continue

            # Skip receiver lines
            if self._is_receiver_line(line):
                continue

            # Stop if we hit content that's clearly not address (like "Sehr geehrte")
            if any(phrase in line.lower() for phrase in ['sehr geehrte', 'dear', 'subject:', 'betreff:', 'angebot nr']):
                break

            # Look for address indicators
            if (any(indicator in line.lower() for indicator in self.address_keywords['street']) or
                any(re.search(pattern, line) for pattern in self.postal_patterns) or
                any(indicator in line.lower() for indicator in self.address_keywords['contact'])):
                
                address_lines.append(line)
                
                # Stop after finding a reasonable amount of address info
                if len(address_lines) >= 4:  # Usually sufficient for address
                    break

        result = '\n'.join(address_lines) if address_lines else None
        
        # Additional safety check - if result is too long, truncate at reasonable point
        if result and len(result) > 300:
            # Try to find a natural break point
            sentences = result.split('. ')
            if len(sentences) > 1:
                result = sentences[0] + '.'
            else:
                # Just truncate at 300 characters
                result = result[:300] + '...'
        
        return result

    def _extract_from_top_lines(self, text: str) -> Optional[str]:
        """Extract address from top lines with better filtering"""
        lines = [line.strip() for line in text.split('\n') if line.strip()][:15]
        address_lines = []

        for line in lines:
            # Stop at content sections
            if any(phrase in line.lower() for phrase in ['sehr geehrte', 'dear', 'subject:', 'betreff:', 'angebot nr']):
                break
                
            if self._is_likely_address_line(line) and not self._is_receiver_line(line):
                address_lines.append(line)
                
                # Limit to reasonable number of lines
                if len(address_lines) >= 3:
                    break

        return '\n'.join(address_lines[:3]) if address_lines else None

    def _extract_using_patterns(self, text: str) -> Optional[str]:
        """Extract address using pattern matching"""
        blocks = self._get_text_blocks(text)
        
        for block in blocks:
            if self._is_receiver_address(block):
                continue

            if len(block) > 300:  # Skip overly long blocks
                continue

            if self._contains_address_patterns(block):
                return block

        return None

    def _is_receiver_line(self, line: str) -> bool:
        """Check if line is part of receiver address"""
        line_lower = line.lower()
        
        receiver_indicators = [
            'an:', 'to:', 'empfÃ¤nger:', 'receiver:', 'bill to:', 'invoice to:',
            'lieferadresse:', 'delivery address:', 'rechnungsadresse:', 'billing address:'
        ]

        for indicator in receiver_indicators:
            if indicator in line_lower:
                return True

        if self.receiver_company and self.receiver_company in line_lower:
            return True

        return False

    def _is_likely_address_line(self, line: str) -> bool:
        """Check if line is likely part of an address"""
        line_lower = line.lower()

        # Exclude contact info and document headers
        exclude_keywords = ['email:', 'tel:', 'fax:', 'web:', 'www.', 'datum:', 'date:',
                            'invoice', 'rechnung', 'angebot', 'betreff:', 'subject:']
        if any(keyword in line_lower for keyword in exclude_keywords):
            return False

        # Include postal codes
        if any(re.search(pattern, line) for pattern in self.postal_patterns):
            return True

        # Include street indicators
        if any(indicator in line_lower for indicator in self.address_keywords['street']):
            return True

        # Include reasonable length lines with mixed content
        if 5 <= len(line) <= 80 and any(c.isdigit() for c in line):
            return True

        return False

    def _select_best_address(self, candidates: List[Tuple[str, str, float]]) -> str:
        """Select best address from candidates - FIXED VERSION"""
        scored_candidates = []

        for method, address, base_score in candidates:
            score = base_score

            # Bonus for postal codes
            if any(re.search(pattern, address) for pattern in self.postal_patterns):
                score += 0.3

            # Bonus for street patterns
            if any(indicator in address.lower() for indicator in self.address_keywords['street']):
                score += 0.2

            # Penalty for too many contact details
            contact_count = sum(1 for keyword in self.address_keywords['contact'] if keyword in address.lower())
            if contact_count > 1:
                score -= 0.2

            # Penalty for very long addresses (likely contains full document text)
            if len(address) > 200:
                score -= 0.5

            scored_candidates.append((address, score, method))

        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if scored_candidates:
            # Return only the address string from the best candidate
            best_address = scored_candidates[0][0]
            print(f"DEBUG: Selected address: '{best_address[:100]}...' (score: {scored_candidates[0][1]:.2f}, method: {scored_candidates})")
            return best_address
        
        return ""

    def _simple_address_extraction(self, text: str) -> str:
        """Simple fallback address extraction"""
        lines = text.split('\n')
        for line in lines[:15]:
            if (any(re.search(pattern, line) for pattern in self.postal_patterns) and
                not self._is_receiver_line(line)):
                return line.strip()
        return ""

    def _get_text_blocks(self, text: str) -> List[str]:
        """Split text into blocks separated by empty lines"""
        lines = [line.strip() for line in text.strip().splitlines()]
        blocks = []
        current_block = []

        for line in lines:
            if not line:
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
            else:
                current_block.append(line)

        if current_block:
            blocks.append('\n'.join(current_block))
        return blocks

    def _contains_address_patterns(self, block: str) -> bool:
        """Check if block contains address patterns"""
        has_postal = any(re.search(pattern, block) for pattern in self.postal_patterns)
        has_street = any(indicator in block.lower() for indicator in self.address_keywords['street'])
        return has_postal or has_street

    def _is_receiver_address(self, block: str) -> bool:
        """Check if block is likely receiver address"""
        return (self.receiver_company and self.receiver_company in block.lower()) or \
               bool(re.search(r'\b(an|to):\s*', block, re.IGNORECASE))

    def extract_fields(self, text: str, language: str = "en") -> Dict[str, str]:
        """Main method to extract all fields from text using enhanced logic with fallback"""
        if not ADVANCED_DEPS_AVAILABLE:
            # Use test.py fallback methods when advanced dependencies are not available
            return extract_fields_with_deepseek_fallback(text)
        
        try:
            # Try original extraction method
            return {
                "date": self.extract_date(text),
                "angebot": self.extract_angebot(text),
                "company_name": self.extract_company_name(text, language),
                "sender_address": self.extract_sender_address(text, language)
            }
        except Exception as e:
            print(f"⚠️ Original extraction failed: {e}, using test.py fallback...")
            return extract_fields_with_deepseek_fallback(text)

# Your convenience functions
def extract_fields_from_pdf(text: str, language: str = "en", receiver_company: str = "") -> Dict[str, str]:
    """Convenience function for extracting fields from PDF text."""
    extractor = PDFFieldExtractor(receiver_company)
    return extractor.extract_fields(text, language)

def process_pdf_documents(pdf_text: str, receiver_company: str = "") -> Dict[str, str]:
    """Specialized function for processing German PDF documents."""
    extractor = PDFFieldExtractor(receiver_company)
    return extractor.extract_fields(pdf_text, language="de")

# =========================
# LAYOUTLMV3 TEXT EXTRACTION (SAME AS BEFORE)
# =========================

def extract_text_with_boxes_from_pdf(pdf_path):
    """Extract words with precise bounding boxes using character-level analysis."""
    all_pages_data = []
    
    try:
        pages = list(extract_pages(str(pdf_path)))
    except Exception as e:
        print(f"Error extracting from {pdf_path}: {str(e)}")
        return all_pages_data
    
    for page_num, page_layout in enumerate(pages):
        words = []
        boxes = []
        width = max(page_layout.width, 1)
        height = max(page_layout.height, 1)
        
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for text_line in element:
                    if hasattr(text_line, '__iter__'):
                        current_word = ''
                        x0, y0, x1, y1 = None, None, None, None
                        
                        for char in text_line:
                            if isinstance(char, LTChar):
                                char_text = char.get_text()
                                
                                if char_text.isspace():
                                    if current_word:
                                        norm_box = [
                                            max(0, min(1000, int(1000 * (x0 / width)))),
                                            max(0, min(1000, int(1000 * (1 - (y1 / height))))),
                                            max(0, min(1000, int(1000 * (x1 / width)))),
                                            max(0, min(1000, int(1000 * (1 - (y0 / height)))))
                                        ]
                                        words.append(current_word.strip())
                                        boxes.append(norm_box)
                                        current_word = ''
                                        x0 = y0 = x1 = y1 = None
                                else:
                                    cx0, cy0, cx1, cy1 = char.bbox
                                    if x0 is None:
                                        x0, y0, x1, y1 = cx0, cy0, cx1, cy1
                                    else:
                                        x0 = min(x0, cx0)
                                        y0 = min(y0, cy0)
                                        x1 = max(x1, cx1)
                                        y1 = max(y1, cy1)
                                    current_word += char_text
                        
                        if current_word.strip():
                            norm_box = [
                                max(0, min(1000, int(1000 * (x0 / width)))),
                                max(0, min(1000, int(1000 * (1 - (y1 / height))))),
                                max(0, min(1000, int(1000 * (x1 / width)))),
                                max(0, min(1000, int(1000 * (1 - (y0 / height)))))
                            ]
                            words.append(current_word.strip())
                            boxes.append(norm_box)
        
        if words:
            all_pages_data.append({
                'words': words,
                'boxes': boxes,
                'page_num': page_num
            })
    
    return all_pages_data

# =========================
# HYBRID EXTRACTOR WITH SOURCE TRACKING
# =========================

class HybridPDFExtractor:
    def __init__(self, model_path: str, receiver_company: str = ""):
        """Initialize hybrid extractor with enhanced positional angebot extraction and source tracking"""
        self.rule_extractor = PDFFieldExtractor(receiver_company)
        self.model_path = model_path
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load LayoutLMv3 model and processor"""
        try:
            self.processor = LayoutLMv3Processor.from_pretrained(self.model_path, apply_ocr=False)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(self.model_path)
            print("âœ… LayoutLMv3 model loaded successfully")
        except Exception as e:
            print(f"âš ï¸ Could not load LayoutLMv3 model: {e}")
            print("Will use enhanced rule-based extraction only")
    
    def extract_with_layoutlmv3(self, pdf_path: Path) -> Dict[str, str]:
        """Extract using LayoutLMv3 model"""
        if not self.model or not self.processor:
            return {"Date": "", "Sender_Company": "", "Sender_Address": "", "Angebot": "", "Table": ""}
        
        try:
            pages_data = extract_text_with_boxes_from_pdf(pdf_path)
            
            if not pages_data:
                return {"Date": "", "Sender_Company": "", "Sender_Address": "", "Angebot": "", "Table": ""}
            
            all_predictions = {
                "DATE": [],
                "COMPANY": [],
                "ADDRESS": [],
                "ANGEBOT": [],
                "TABLE": []
            }
            
            # Process first page only for efficiency
            for page_data in pages_data[:1]:
                words = page_data['words']
                boxes = page_data['boxes']
                
                if not words:
                    continue
                
                # Convert first page to image
                try:
                    pdf_images = convert_from_path(str(pdf_path), dpi=150, first_page=1, last_page=1)
                    if pdf_images:
                        image = pdf_images[0].resize((224, 224))
                    else:
                        image = Image.new('RGB', (224, 224), color='white')
                except:
                    image = Image.new('RGB', (224, 224), color='white')
                
                try:
                    encoding = self.processor(
                        image,
                        words,
                        boxes=boxes,
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(**encoding)
                    
                    predictions = outputs.logits.argmax(-1).squeeze().tolist()
                    if isinstance(predictions, int):
                        predictions = [predictions]
                    
                    # Extract entities
                    current_entity = []
                    current_type = None
                    
                    for word, pred_id in zip(words, predictions[:len(words)]):
                        label = ID2LABEL.get(pred_id, "O")
                        
                        if label.startswith("B-"):
                            if current_entity and current_type:
                                all_predictions[current_type].append(" ".join(current_entity))
                            current_type = label[2:]
                            current_entity = [word]
                        elif label.startswith("I-") and current_type == label[2:]:
                            current_entity.append(word)
                        else:
                            if current_entity and current_type:
                                all_predictions[current_type].append(" ".join(current_entity))
                            current_entity = []
                            current_type = None
                    
                    if current_entity and current_type:
                        all_predictions[current_type].append(" ".join(current_entity))
                        
                except Exception as e:
                    print(f"Error processing with LayoutLMv3: {e}")
                    continue
            
            # Aggregate predictions
            result = {
                "Date": all_predictions["DATE"][0] if all_predictions["DATE"] else "",
                "Sender_Company": all_predictions["COMPANY"] if all_predictions["COMPANY"] else "",
                "Sender_Address": " ".join(all_predictions["ADDRESS"][:2]) if all_predictions["ADDRESS"] else "",
                "Angebot": all_predictions["ANGEBOT"] if all_predictions["ANGEBOT"] else "",
                "Table": " ".join(all_predictions["TABLE"][:2]) if all_predictions["TABLE"] else ""
            }
            
            return result
            
        except Exception as e:
            print(f"LayoutLMv3 extraction failed for {pdf_path}: {e}")
            return {"Date": "", "Sender_Company": "", "Sender_Address": "", "Angebot": "", "Table": ""}
    
    def extract_with_rules(self, text: str, language: str = "de") -> Dict[str, str]:
        """Extract using enhanced rule-based method with positional angebot extraction"""
        return self.rule_extractor.extract_fields(text, language)
    
    def extract_hybrid(self, pdf_path: Path, text: str, language: str = "de") -> Dict[str, str]:
        """
        Hybrid extraction with SOURCE TRACKING for each field
        """
        print(f"\nðŸ”„ Processing {pdf_path.name} with source tracking...")
        
        # Method 1: LayoutLMv3 extraction
        layoutlm_results = self.extract_with_layoutlmv3(pdf_path)
        print(f"LayoutLMv3 results: {layoutlm_results}")
        
        # Method 2: Enhanced rule-based extraction
        rule_results = self.extract_with_rules(text, language)
        print(f"Enhanced rule-based results: {rule_results}")
        
        # Method 3: Intelligent fusion WITH SOURCE TRACKING
        final_results = {}
        sources = {}  # Track which method was chosen for each field
        
        fields_mapping = {
            "Date": "date",
            "Sender_Company": "company_name", 
            "Sender_Address": "sender_address",
            "Angebot": "angebot"
        }
        
        for layoutlm_key, rule_key in fields_mapping.items():
            layoutlm_value = safe_str(layoutlm_results.get(layoutlm_key, "")).strip()
            rule_value = safe_str(rule_results.get(rule_key, "")).strip()
            
            # Enhanced decision logic with SOURCE TRACKING
            if layoutlm_key == "Date":
                if rule_value:
                    final_results[layoutlm_key] = rule_value
                    sources[layoutlm_key] = "Rule-Based"
                    print(f"âœ… Date: Using Rule-Based - '{rule_value}'")
                else:
                    final_results[layoutlm_key] = layoutlm_value
                    sources[layoutlm_key] = "Model"
                    print(f"âœ… Date: Using Model - '{layoutlm_value}'")
                    
            elif layoutlm_key == "Angebot":
                # Validate both results
                valid_rule = rule_value and self.rule_extractor._valid_angebot(rule_value)
                valid_layoutlm = layoutlm_value and self.rule_extractor._valid_angebot(layoutlm_value)
                
                if valid_rule:
                    final_results[layoutlm_key] = rule_value
                    sources[layoutlm_key] = "Rule-Based (Positional)"
                    print(f"âœ… Angebot: Using Rule-Based (Positional) - '{rule_value}'")
                elif valid_layoutlm:
                    final_results[layoutlm_key] = layoutlm_value
                    sources[layoutlm_key] = "Model"
                    print(f"âœ… Angebot: Using Model - '{layoutlm_value}'")
                else:
                    final_results[layoutlm_key] = ""
                    sources[layoutlm_key] = "None"
                    print(f"âŒ Angebot: No valid result found")
                    
            elif layoutlm_key == "Sender_Company":
                if rule_value and len(rule_value) > 3:
                    final_results[layoutlm_key] = rule_value
                    sources[layoutlm_key] = "Rule-Based"
                    print(f"âœ… Company: Using Rule-Based - '{rule_value}'")
                elif layoutlm_value and len(layoutlm_value) > 3:
                    final_results[layoutlm_key] = layoutlm_value
                    sources[layoutlm_key] = "Model"
                    print(f"âœ… Company: Using Model - '{layoutlm_value}'")
                else:
                    final_results[layoutlm_key] = rule_value or layoutlm_value
                    sources[layoutlm_key] = "Rule-Based" if rule_value else "Model"
                    chosen_value = rule_value or layoutlm_value
                    chosen_method = "Rule-Based" if rule_value else "Model"
                    print(f"âœ… Company: Using {chosen_method} (fallback) - '{chosen_value}'")
                    
            elif layoutlm_key == "Sender_Address":
                if rule_value and len(rule_value) > 15:
                    final_results[layoutlm_key] = rule_value
                    sources[layoutlm_key] = "Rule-Based"
                    print(f"âœ… Address: Using Rule-Based - '{rule_value[:50]}...'")
                elif layoutlm_value and len(layoutlm_value) > 15:
                    final_results[layoutlm_key] = layoutlm_value
                    sources[layoutlm_key] = "Model"
                    print(f"âœ… Address: Using Model - '{layoutlm_value[:50]}...'")
                else:
                    final_results[layoutlm_key] = rule_value or layoutlm_value
                    sources[layoutlm_key] = "Rule-Based" if rule_value else "Model"
                    chosen_value = rule_value or layoutlm_value
                    chosen_method = "Rule-Based" if rule_value else "Model"
                    print(f"âœ… Address: Using {chosen_method} (fallback) - '{chosen_value[:50]}...'")
        
        # Add table extraction from LayoutLMv3
        final_results["Table"] = safe_str(layoutlm_results.get("Table", ""))
        final_results["Table_File"] = f"{pdf_path.stem}.xlsx"
        sources["Table"] = "Model"  # Table extraction is always from model
        
        # Add source information for each field
        for field in fields_mapping.keys():
            final_results[f"{field}_Source"] = sources.get(field, "Unknown")
        
        # Add source for Table
        final_results["Table_Source"] = sources.get("Table", "Model")
        
        print(f"âœ… Enhanced hybrid results with source tracking: {final_results}")
        return final_results

# =========================
# TESTING FUNCTIONS (UPDATED FOR SOURCE TRACKING)
# =========================

def test_single_pdf():
    """Test extraction on a single PDF with source tracking"""
    print("\n" + "="*60)
    print("ðŸ§ª SINGLE PDF TEST MODE - WITH SOURCE TRACKING")
    print("="*60)
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"âŒ No PDF files found in: {pdf_folder}")
        return None
    
    print("Available PDFs:")
    for i, pdf in enumerate(pdf_files[:10]):
        print(f"   {i+1}. {pdf.name}")
    
    test_pdf_path = pdf_files[0]
    print(f"\nðŸ” Testing PDF: {test_pdf_path.name}")
    
    # Initialize hybrid extractor
    model_path = output_dir / "best_model"
    if not model_path.exists():
        model_path = local_model_path
    
    hybrid_extractor = HybridPDFExtractor(str(model_path), receiver_company="")
    
    # Extract plain text for rule-based processing
    try:
        plain_text = extract_text(str(test_pdf_path))
        print(f"\nðŸ“„ Extracted Text Preview (first 1000 characters):")
        print("-" * 50)
        print(plain_text[:1000])
        print("-" * 50)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None
    
    # Test enhanced rule-based extraction first
    print(f"\nðŸ”§ Testing positional angebot extraction...")
    rule_only_results = process_pdf_documents(plain_text, receiver_company="")
    print("Enhanced Rule-based Results (with positional logic):")
    for field, value in rule_only_results.items():
        value_str = safe_str(value)
        status = "âœ“" if value_str and value_str.strip() else "âœ—"
        if field == "angebot" and value_str:
            # Show validation result
            extractor = PDFFieldExtractor()
            is_valid = extractor._valid_angebot(value_str)
            status = "âœ… Valid" if is_valid else "âŒ Invalid"
        print(f"  {status} {field:15}: {value_str}")
    
    # Run full hybrid extraction with source tracking
    print(f"\nðŸ”„ Running full hybrid extraction with source tracking...")
    result = hybrid_extractor.extract_hybrid(test_pdf_path, plain_text, language="de")
    
    # Display results with sources
    print(f"\nðŸ“Š EXTRACTION RESULTS WITH SOURCE TRACKING:")
    print("=" * 60)
    
    fields_to_display = ["Date", "Sender_Company", "Sender_Address", "Angebot"]
    for field in fields_to_display:
        value = safe_str(result.get(field, ""))
        source = result.get(f"{field}_Source", "Unknown")
        status = "âœ“" if value and value.strip() else "âœ—"
        print(f"{status} {field:15}: {value} [{source}]")
    
    # Save test results with sources
    test_output = output_dir / f"test_result_with_sources_{test_pdf_path.stem}.json"
    json_safe_result = {k: safe_str(v) for k, v in result.items()}
    
    with open(test_output, 'w') as f:
        json.dump(json_safe_result, f, indent=2, ensure_ascii=False)
    print(f"\nðŸ’¾ Test results with sources saved to: {test_output}")
    
    # Quality assessment
    filled_fields = sum(1 for k, v in result.items() 
                       if not k.endswith("_Source") and k != "Table_File" and safe_str(v).strip())
    total_fields = len([k for k in result.keys() if not k.endswith("_Source") and k != "Table_File"])
    quality_score = (filled_fields / total_fields) * 100
    
    print(f"\nðŸ“Š QUALITY ASSESSMENT:")
    print(f"Filled fields: {filled_fields}/{total_fields}")
    print(f"Quality score: {quality_score:.1f}%")
    
    # Special focus on Angebot validation
    angebot_value = safe_str(result.get("Angebot", ""))
    angebot_source = result.get("Angebot_Source", "")
    if angebot_value:
        extractor = PDFFieldExtractor()
        is_valid_angebot = extractor._valid_angebot(angebot_value)
        print(f"Angebot validation: {'âœ… Valid' if is_valid_angebot else 'âŒ Invalid'} (Source: {angebot_source})")
    else:
        print(f"Angebot: âŒ Not found (Source: {angebot_source})")
    
    if quality_score >= 75:
        print("ðŸŽ‰ Excellent extraction quality with source tracking!")
    elif quality_score >= 50:
        print("ðŸ‘ Good extraction quality!")
    else:
        print("âš ï¸ Consider checking PDF text quality or adjusting parameters")
        
    return result

def extract_text_from_pdf_simple(pdf_path: Path) -> str:
    """Simple text extraction for rule-based processing"""
    try:
        return extract_text(str(pdf_path))
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def main_hybrid():
    print("\n" + "="*70)
    print("ðŸš€ ENHANCED HYBRID PIPELINE WITH SOURCE TRACKING")
    print("="*70)
    
    # Initialize hybrid extractor
    model_path = output_dir / "best_model"
    if not model_path.exists():
        model_path = local_model_path
    
    hybrid_extractor = HybridPDFExtractor(str(model_path), receiver_company="")
    
    # Process all PDFs
    predictions = []
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    
    print(f"\nðŸ“„ Processing {len(pdf_files)} PDFs with source tracking...")
    
    for pdf_file in tqdm(pdf_files, desc="Extracting"):
        plain_text = extract_text_from_pdf_simple(pdf_file)
        result = hybrid_extractor.extract_hybrid(pdf_file, plain_text, language="de")
        
        # Convert any list values to strings for Excel compatibility
        safe_result = {k: safe_str(v) for k, v in result.items()}
        
        predictions.append({
            "filename": pdf_file.name,
            "Date": safe_result.get("Date", ""),
            "Date_Source": safe_result.get("Date_Source", ""),
            "Sender_Company": safe_result.get("Sender_Company", ""),
            "Sender_Company_Source": safe_result.get("Sender_Company_Source", ""),
            "Sender_Address": safe_result.get("Sender_Address", ""),
            "Sender_Address_Source": safe_result.get("Sender_Address_Source", ""),
            "Angebot": safe_result.get("Angebot", ""),
            "Angebot_Source": safe_result.get("Angebot_Source", ""),
            "Table": safe_result.get("Table", ""),
            "Table_Source": safe_result.get("Table_Source", ""),
            "Table_File": safe_result.get("Table_File", f"{pdf_file.stem}.xlsx")
        })
    
    # Save predictions with source tracking
    output_file = output_dir / "predictions_with_source_tracking.xlsx"
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_excel(output_file, index=False)
    print(f"\nâœ… Enhanced hybrid predictions (with source tracking) saved to: {output_file}")
    
    # Enhanced analytics with source tracking
    print(f"\nðŸ“Š SOURCE TRACKING ANALYSIS:")
    print(f"Total PDFs processed: {len(df_predictions)}")
    
    # Analyze source distribution for each field
    fields_to_analyze = ["Date", "Sender_Company", "Sender_Address", "Angebot"]
    
    for field in fields_to_analyze:
        source_col = f"{field}_Source"
        if source_col in df_predictions.columns:
            source_counts = df_predictions[source_col].value_counts()
            print(f"\n{field} Sources:")
            for source, count in source_counts.items():
                percentage = (count / len(df_predictions)) * 100
                print(f"  â€¢ {source}: {count} ({percentage:.1f}%)")
    
    # Calculate success metrics with source breakdown
    filled_predictions = df_predictions[
        (df_predictions['Date'] != '') | 
        (df_predictions['Sender_Company'] != '') | 
        (df_predictions['Sender_Address'] != '') | 
        (df_predictions['Angebot'] != '')
    ]
    
    # Special focus on Angebot extraction success
    valid_angebot_count = 0
    extractor = PDFFieldExtractor()
    
    for _, row in df_predictions.iterrows():
        angebot_value = row['Angebot']
        if angebot_value and extractor._valid_angebot(angebot_value):
            valid_angebot_count += 1
    
    print(f"\nðŸ“ˆ OVERALL SUCCESS METRICS:")
    print(f"Total PDFs processed: {len(df_predictions)}")
    print(f"PDFs with extracted entities: {len(filled_predictions)}")
    print(f"PDFs with VALID Angebot: {valid_angebot_count}")
    print(f"Overall success rate: {len(filled_predictions)/len(df_predictions)*100:.1f}%")
    print(f"Valid Angebot extraction rate: {valid_angebot_count/len(df_predictions)*100:.1f}%")
    
    # Show sample results with sources
    print(f"\nðŸ“‹ SAMPLE RESULTS WITH SOURCE TRACKING (first 5 PDFs):")
    for i, row in df_predictions.head().iterrows():
        angebot_value = row['Angebot']
        angebot_source = row['Angebot_Source']
        if angebot_value:
            is_valid = extractor._valid_angebot(angebot_value)
            angebot_status = "âœ… Found (valid)" if is_valid else "âš ï¸ Found (invalid)"
        else:
            angebot_status = "âŒ Not found"
            
        print(f"\n{i+1}. {row['filename']} - {angebot_status}")
        print(f"   Date: {row['Date']} [{row['Date_Source']}]")
        print(f"   Company: {row['Sender_Company']} [{row['Sender_Company_Source']}]")
        address_display = row['Sender_Address'][:50] + "..." if len(row['Sender_Address']) > 50 else row['Sender_Address']
        print(f"   Address: {address_display} [{row['Sender_Address_Source']}]")
        print(f"   Angebot: {row['Angebot']} [{angebot_source}]")
    
    print("\n" + "="*70)
    print("ðŸŽ‰ SOURCE TRACKING EXTRACTION COMPLETE!")
    print("Excel output now includes source information for each field!")
    print("="*70)

# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Test single PDF (with source tracking)")
    print("2. Process all PDFs (with source tracking)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_single_pdf()
    elif choice == "2":
        main_hybrid()
    else:
        print("Invalid choice. Testing single PDF by default.")
        test_single_pdf()