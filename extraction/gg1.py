import re
import logging
import sqlite3
import hashlib
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import torch
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'


try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logger = logging.getLogger(__name__)

class AdaptiveFixedTableExtractor:
    """Complete fixed extractor with adaptive learning, column alignment and 20-char limit."""
    
    def __init__(self, debug: bool = False, learning_db_path: str = "table_learning.db"):
        self.debug = debug
        self.char_limit = 20  # CHARACTER LIMIT per cell
        self.learning_db_path = learning_db_path
        
        # Initialize learning database
        self.init_learning_database()
        
        # Load any previously learned patterns
        self.load_learned_patterns()
        
        # COMPREHENSIVE: Target parameters with extensive synonyms
        self.target_parameters = {
            
            'pos': ['pos','Pos' 'position', 'no.', '#', 'nummer', 'item no', 'sr no'],

            'artikel_nr': [
                 'nr', 'item', 'artikel', 'art-nr', 'art.nr', 'position', '#', 
                'item-nr', 'item nr', 'product nr', 'product-nr', 'produktnummer', 'teil-nr',
                'material-nr', 'code', 'produktcode', 'itemcode', 'stock nr', 'reference',
                'ref', 'article', 'stock number', 'catalog nr', 'sku', 'id', 'lfd', 'lfd.','paket','Nr.'
            ],
            'beschreibung': [
                'beschreibung', 'description', 'bezeichnung', 'artikel', 'material', 'product',
                'artikelbezeichnung', 'materialbezeichnung', 'produktname', 'produktbezeichnung',
                'leistung', 'leistungsbeschreibung', 'name', 'titel', 'text', 'inhalt',
                'item description', 'product description', 'service description', 'work description',
                'work package', 'task', 'service', 'goods', 'commodity', 'specification',
                'details', 'subject', 'content', 'work', 'activity', 'desc', 'short description','inhalt',"Frage"
            ],
            'einzelpreis': [
                'einzelpreis', 'preis', 'price', 'rate', 'e-preis', 'unit price', 'stÃ¼ckpreis',
                'einzel-preis', 'ep', 'nettpreis', 'listenpreis', 'verkaufspreis', 'bruttopreis',
                'grundpreis', 'basispreis', 'tarif', 'satz', 'kostensatz', 'stundensatz',
                'cost', 'fee', 'charge', 'list price', 'selling price', 'net price',
                'gross price', 'base price', 'price per unit', 'cost per unit', 'hourly rate'
            ],
            'gesamtpreis': [
                'gesamtpreis', 'total', 'summe', 'amount', 'g-preis', 'gesamt', 'gesamtbetrag',
                'gesamt-preis', 'gesamtsumme', 'totalsumme', 'endsumme', 'betrag', 'wert',
                'gesamtwert', 'nettobetrag', 'bruttobetrag', 'rechnungsbetrag', 'total price',
                'total amount', 'total value', 'grand total', 'sum', 'total cost', 'line total',
                'subtotal', 'net amount', 'gross amount', 'final amount', 'invoice amount'
            ],
            'menge': [
                'menge', 'qty', 'anzahl', 'quantity', 'stk', 'pcs', 'anz', 'stÃ¼ck',
                'quantitÃ¤t', 'anzahl stÃ¼ck', 'stÃ¼ckzahl', 'einheiten', 'einheit', 'zahl',
                'number', 'count', 'pieces', 'units', 'unit', 'volume', 'size', 'total',
                'lot', 'batch', 'portion', 'share', 'part', 'number of', 'no of', 'nos'
            ]
        }
        
        # Data validation patterns
        self.data_patterns = {
            'price': re.compile(r'\d{1,8}[.,]\d{1,4}(?:\s*â‚¬|\s*EUR)?', re.IGNORECASE),
            'quantity': re.compile(r'^\d{1,6}([.,]\d{1,4})?\s*(st|stk|pcs|kg|g|m|stÃ¼ck|pieces|units)?$', re.IGNORECASE),
            'artikel_number': re.compile(r'^[A-Za-z0-9\-\.\/]{1,20}$'),
            'reject_lines': re.compile(r'^(seite\s|page\s|confidential|proprietary|tel:|fax:|email:|www\.)', re.IGNORECASE)
        }
        
    # Define the local path (using raw string for Windows compatibility)
    local_model_path = r"C:\Users\I702209\Documents\Project\model"

    # Load processor offline from local path
    processor = LayoutLMv3Processor.from_pretrained(local_model_path)

    # Load model offline from local path
    model = LayoutLMv3ForTokenClassification.from_pretrained(local_model_path)

    # Your existing settings (adjust as needed for your fine-tuning)
    TARGET_CLASS_INDEX = 2  # e.g., for 'RELEVANT_TABLE'
    CONFIDENCE_THRESHOLD = 0  # Adjustable

     
        
        

    def init_learning_database(self):
        """Initialize database for storing corrections and learned patterns."""
        Path(self.learning_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.learning_db_path) as conn:
            # Table corrections storage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS table_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    table_index INTEGER NOT NULL,
                    correction_type TEXT NOT NULL,
                    row_index INTEGER,
                    col_index INTEGER,
                    original_value TEXT,
                    corrected_value TEXT NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.95
                )
            """)
            
            # Learned patterns storage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    param_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 1,
                    success_count INTEGER DEFAULT 1,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(param_type, pattern)
                )
            """)
            
            # Document processing history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT UNIQUE NOT NULL,
                    file_name TEXT,
                    first_processed TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_processed TEXT DEFAULT CURRENT_TIMESTAMP,
                    processing_count INTEGER DEFAULT 1,
                    correction_count INTEGER DEFAULT 0
                )
            """)
            
            # Quality improvement tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_improvements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    original_quality REAL,
                    improved_quality REAL,
                    improvement_type TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        if self.debug:
            print("âœ… Learning database initialized")

    def load_learned_patterns(self):
        """Load previously learned patterns from database."""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                cursor = conn.execute("""
                    SELECT param_type, pattern, confidence, usage_count
                    FROM learned_patterns
                    WHERE confidence >= 0.5
                    ORDER BY usage_count DESC, confidence DESC
                """)
                
                loaded_count = 0
                for param_type, pattern, confidence, usage_count in cursor.fetchall():
                    if param_type in self.target_parameters:
                        if pattern not in self.target_parameters[param_type]:
                            self.target_parameters[param_type].append(pattern)
                            loaded_count += 1
                
                if self.debug and loaded_count > 0:
                    print(f"âœ… Loaded {loaded_count} learned patterns")
                    
        except Exception as e:
            if self.debug:
                print(f"âš ï¸ Could not load learned patterns: {e}")

    def extract_tables_with_learning(self, file_path: str) -> List[Dict]:
        """Main extraction method with adaptive learning applied."""
        
        if not os.path.exists(file_path):
            if self.debug:
                print(f"âŒ File not found: {file_path}")
            return []
        
        if pdfplumber is None:
            if self.debug:
                print("âŒ pdfplumber not available!")
            return []
        
        # Generate document hash for learning
        with open(file_path, 'rb') as f:
            document_hash = hashlib.sha256(f.read()).hexdigest()[:32]
        
        # Update document history
        self.update_document_history(document_hash, os.path.basename(file_path))
        
        if self.debug:
            print(f"{'='*80}")
            print(f"ADAPTIVE FIXED TABLE EXTRACTION")
            print(f"File: {os.path.basename(file_path)}")
            print(f"Document Hash: {document_hash}")
            print(f"Character limit per cell: {self.char_limit}")
            print(f"{'='*80}")
        
        all_tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    if self.debug:
                        print(f"\n--- PROCESSING PAGE {page_num} ---")
                    
                    # Extract structured data
                    page_data = self._extract_structured_data(page, page_num)
                    
                    # Detect and format tables
                    tables = self._detect_and_format_tables(page_data, page_num)
                    
                    all_tables.extend(tables)
            # Step: Filter candidate tables with LayoutLMv3
            #all_tables = self.filter_tables_with_layoutlmv3(all_tables)
            # Apply learned corrections
            all_tables = self.apply_learned_corrections(document_hash, all_tables)
            

            # Add document hash to each table for tracking
            for table in all_tables:
                table['document_hash'] = document_hash
                table['extraction_method'] = 'adaptive_fixed_20char'
                    
        except Exception as e:
            if self.debug:
                print(f"âŒ Extraction failed: {e}")
            return []
        
        if self.debug:
            print(f"\nðŸ EXTRACTION COMPLETE: Found {len(all_tables)} tables")
        
        return all_tables

    def apply_learned_corrections(self, document_hash: str, tables: List[Dict]) -> List[Dict]:
        """Apply previously learned corrections to extracted tables."""
        corrections = self.get_saved_corrections(document_hash)
        
        if not corrections:
            return tables
        
        if self.debug:
            print(f"âœ… Applying learned corrections for document {document_hash[:8]}...")
        
        applied_count = 0
        
        for table_idx, table in enumerate(tables):
            table_corrections = corrections.get(str(table_idx), {})
            
            # Apply header corrections
            if 'header' in table_corrections:
                for col_idx, corrected_value in table_corrections['header'].items():
                    if int(col_idx) < len(table.get('headers', [])):
                        original_value = table['headers'][int(col_idx)]
                        table['headers'][int(col_idx)] = corrected_value
                        applied_count += 1
                        
                        if self.debug:
                            print(f"  Header correction: '{original_value}' â†’ '{corrected_value}'")
            
            # Apply cell corrections
            if 'cell' in table_corrections:
                table_data = table.get('table_data', [])
                for cell_key, corrected_value in table_corrections['cell'].items():
                    try:
                        row_idx, col_idx = map(int, cell_key.split('_'))
                        if row_idx < len(table_data) and col_idx < len(table_data[row_idx]):
                            original_value = table_data[row_idx][col_idx]
                            table_data[row_idx][col_idx] = corrected_value
                            applied_count += 1
                            
                            if self.debug:
                                print(f"  Cell correction [{row_idx},{col_idx}]: '{original_value}' â†’ '{corrected_value}'")
                    except (ValueError, IndexError):
                        continue
        
        if self.debug and applied_count > 0:
            print(f"âœ… Applied {applied_count} total corrections")
        
        return tables

    def save_user_corrections(self, document_hash: str, corrections: Dict[str, Any]):
        """Save user corrections for learning."""
        if not corrections:
            return
        
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                saved_count = 0
                
                for table_idx, table_corrections in corrections.items():
                    for correction_type, changes in table_corrections.items():
                        for key, correction_data in changes.items():
                            if isinstance(correction_data, dict):
                                original = correction_data.get('original', '')
                                corrected = correction_data.get('corrected', '')
                            else:
                                original = ''
                                corrected = str(correction_data)
                            
                            if corrected:
                                # Parse row/col indices for cell corrections
                                row_idx = col_idx = None
                                if correction_type == 'cell' and '_' in key:
                                    try:
                                        row_idx, col_idx = map(int, key.split('_'))
                                    except ValueError:
                                        pass
                                elif correction_type == 'header':
                                    try:
                                        col_idx = int(key)
                                    except ValueError:
                                        pass
                                
                                conn.execute("""
                                    INSERT INTO table_corrections 
                                    (document_hash, table_index, correction_type, row_index, col_index, 
                                     original_value, corrected_value, timestamp)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (document_hash, int(table_idx), correction_type, row_idx, col_idx,
                                      original, corrected, datetime.now().isoformat()))
                                saved_count += 1
                
                conn.commit()
                
                # Update document correction count
                conn.execute("""
                    UPDATE document_history 
                    SET correction_count = correction_count + ?, last_processed = ?
                    WHERE document_hash = ?
                """, (saved_count, datetime.now().isoformat(), document_hash))
                conn.commit()
                
                if self.debug:
                    print(f"âœ… Saved {saved_count} corrections to learning database")
            
            # Learn from corrections
            self.learn_from_corrections(corrections)
            
        except Exception as e:
            if self.debug:
                print(f"âŒ Error saving corrections: {e}")

    def learn_from_corrections(self, corrections: Dict[str, Any]):
        """Learn new patterns from user corrections."""
        learning_count = 0
        
        try:
            for table_idx, table_corrections in corrections.items():
                # Learn from header corrections
                if 'header' in table_corrections:
                    for col_idx, correction_data in table_corrections['header'].items():
                        if isinstance(correction_data, dict):
                            original = correction_data.get('original', '')
                            corrected = correction_data.get('corrected', '')
                        else:
                            corrected = str(correction_data)
                            original = ''
                        
                        if corrected and self.learn_header_pattern(original, corrected):
                            learning_count += 1
                
                # Learn from cell patterns (for future enhancement)
                if 'cell' in table_corrections:
                    for cell_key, correction_data in table_corrections['cell'].items():
                        if isinstance(correction_data, dict):
                            corrected = correction_data.get('corrected', '')
                        else:
                            corrected = str(correction_data)
                        
                        if corrected and self.learn_cell_pattern(corrected):
                            learning_count += 1
            
            if self.debug and learning_count > 0:
                print(f"ðŸ§  Learned {learning_count} new patterns")
                
        except Exception as e:
            if self.debug:
                print(f"âŒ Error learning from corrections: {e}")

    def learn_header_pattern(self, original_header: str, corrected_header: str) -> bool:
        """Learn new header patterns from corrections."""
        if not corrected_header.strip():
            return False
        
        corrected_lower = corrected_header.lower().strip()
        
        # Determine which parameter type this should map to
        best_param_type = None
        max_matches = 0
        
        for param_type, patterns in self.target_parameters.items():
            matches = sum(1 for pattern in patterns if pattern in corrected_lower or corrected_lower in pattern)
            if matches > max_matches:
                max_matches = matches
                best_param_type = param_type
        
        # If we found a good match and the pattern is new
        if best_param_type and corrected_lower not in self.target_parameters[best_param_type]:
            self.target_parameters[best_param_type].append(corrected_lower)
            self.save_learned_pattern(best_param_type, corrected_lower)
            
            if self.debug:
                print(f"  ðŸ§  Learned header pattern: '{corrected_lower}' â†’ {best_param_type}")
            return True
        
        return False

    def learn_cell_pattern(self, corrected_value: str) -> bool:
        """Learn patterns from cell corrections (for future enhancement)."""
        # This could be enhanced to learn data formatting patterns
        # For now, we just validate the correction
        return bool(corrected_value.strip())

    def save_learned_pattern(self, param_type: str, pattern: str, confidence: float = 0.8):
        """Save learned pattern to database."""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learned_patterns 
                    (param_type, pattern, confidence, usage_count, success_count, timestamp)
                    VALUES (?, ?, ?, 1, 1, ?)
                """, (param_type, pattern, confidence, datetime.now().isoformat()))
                conn.commit()
                
        except Exception as e:
            if self.debug:
                print(f"âŒ Error saving learned pattern: {e}")

    def get_saved_corrections(self, document_hash: str) -> Dict[str, Dict]:
        """Retrieve saved corrections for a document."""
        corrections = defaultdict(lambda: defaultdict(dict))
        
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                cursor = conn.execute("""
                    SELECT table_index, correction_type, row_index, col_index, original_value, corrected_value
                    FROM table_corrections
                    WHERE document_hash = ?
                    ORDER BY timestamp DESC
                """, (document_hash,))
                
                for row in cursor.fetchall():
                    table_idx, correction_type, row_idx, col_idx, original, corrected = row
                    
                    if correction_type == 'header' and col_idx is not None:
                        corrections[str(table_idx)]['header'][str(col_idx)] = corrected
                    elif correction_type == 'cell' and row_idx is not None and col_idx is not None:
                        corrections[str(table_idx)]['cell'][f"{row_idx}_{col_idx}"] = corrected
                
        except Exception as e:
            if self.debug:
                print(f"âš ï¸ Could not load corrections: {e}")
        
        return dict(corrections)

    def update_document_history(self, document_hash: str, file_name: str):
        """Update document processing history."""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                # Check if document exists
                cursor = conn.execute("SELECT processing_count FROM document_history WHERE document_hash = ?", (document_hash,))
                result = cursor.fetchone()
                
                if result:
                    # Update existing record
                    conn.execute("""
                        UPDATE document_history 
                        SET last_processed = ?, processing_count = processing_count + 1
                        WHERE document_hash = ?
                    """, (datetime.now().isoformat(), document_hash))
                else:
                    # Insert new record
                    conn.execute("""
                        INSERT INTO document_history (document_hash, file_name)
                        VALUES (?, ?)
                    """, (document_hash, file_name))
                
                conn.commit()
                
        except Exception as e:
            if self.debug:
                print(f"âš ï¸ Could not update document history: {e}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                cursor = conn.cursor()
                
                # Total corrections
                cursor.execute("SELECT COUNT(*) FROM table_corrections")
                total_corrections = cursor.fetchone()[0]
                
                # Learned patterns
                cursor.execute("SELECT COUNT(*) FROM learned_patterns")
                learned_patterns = cursor.fetchone()[0]
                
                # Documents with corrections
                cursor.execute("SELECT COUNT(DISTINCT document_hash) FROM table_corrections")
                docs_with_corrections = cursor.fetchone()[0]
                
                # Total processed documents
                cursor.execute("SELECT COUNT(*) FROM document_history")
                total_documents = cursor.fetchone()[0]
                
                # Average corrections per document
                avg_corrections = total_corrections / max(docs_with_corrections, 1)
                
                # Most corrected parameter types
                cursor.execute("""
                    SELECT correction_type, COUNT(*) as count
                    FROM table_corrections
                    GROUP BY correction_type
                    ORDER BY count DESC
                """)
                correction_types = cursor.fetchall()
                
                # Pattern success rates
                cursor.execute("""
                    SELECT param_type, AVG(confidence), COUNT(*) as pattern_count
                    FROM learned_patterns
                    GROUP BY param_type
                    ORDER BY pattern_count DESC
                """)
                pattern_stats = cursor.fetchall()
                
                return {
                    'total_corrections': total_corrections,
                    'learned_patterns': learned_patterns,
                    'documents_with_corrections': docs_with_corrections,
                    'total_documents_processed': total_documents,
                    'average_corrections_per_document': avg_corrections,
                    'correction_types': correction_types,
                    'pattern_statistics': pattern_stats,
                    'learning_rate': docs_with_corrections / max(total_documents, 1) * 100
                }
                
        except Exception as e:
            if self.debug:
                print(f"âŒ Error getting learning statistics: {e}")
            return self._get_empty_learning_stats()

    def _get_empty_learning_stats(self) -> Dict[str, Any]:
        """Return empty learning statistics."""
        return {
            'total_corrections': 0,
            'learned_patterns': 0,
            'documents_with_corrections': 0,
            'total_documents_processed': 0,
            'average_corrections_per_document': 0,
            'correction_types': [],
            'pattern_statistics': [],
            'learning_rate': 0
        }

    def clear_learning_data(self, document_hash: Optional[str] = None):
        """Clear learning data - optionally for specific document."""
        try:
            with sqlite3.connect(self.learning_db_path) as conn:
                if document_hash:
                    # Clear data for specific document
                    conn.execute("DELETE FROM table_corrections WHERE document_hash = ?", (document_hash,))
                    conn.execute("DELETE FROM document_history WHERE document_hash = ?", (document_hash,))
                    conn.execute("DELETE FROM quality_improvements WHERE document_hash = ?", (document_hash,))
                    if self.debug:
                        print(f"âœ… Cleared learning data for document {document_hash[:8]}...")
                else:
                    # Clear all learning data
                    conn.execute("DELETE FROM table_corrections")
                    conn.execute("DELETE FROM learned_patterns")
                    conn.execute("DELETE FROM document_history")
                    conn.execute("DELETE FROM quality_improvements")
                    if self.debug:
                        print("âœ… Cleared all learning data")
                
                conn.commit()
                
        except Exception as e:
            if self.debug:
                print(f"âŒ Error clearing learning data: {e}")

    # ==================================================================================
    # ORIGINAL EXTRACTION METHODS (with minor enhancements for learning integration)
    # ==================================================================================

    def extract_tables_fixed(self, file_path: str) -> List[Dict]:
        """Legacy method - redirects to learning-enabled extraction."""
        return self.extract_tables_with_learning(file_path)

    def _extract_structured_data(self, page, page_num: int) -> Dict:
        """IMPROVED: Extract data with better structure detection."""
        try:
            words = page.extract_words(extra_attrs=['fontname', 'size'])
        except:
            words = page.extract_words() if hasattr(page, "extract_words") else []
        
        if not words:
            return {'lines': [], 'word_groups': []}
        
        if self.debug:
            print(f"  Words extracted: {len(words)}")
        
        # FIXED: Better Y-coordinate clustering for proper row alignment
        y_tolerance = 2.5  # Reduced tolerance for better precision
        lines_dict = defaultdict(list)
        
        for word in words:
            # IMPROVED: More precise Y-coordinate grouping
            y_key = round(word["top"] / y_tolerance) * y_tolerance
            
            word_info = {
                'text': word["text"].strip(),
                'x0': word["x0"],
                'x1': word["x1"],
                'y0': word.get("top", 0),
                'y1': word.get("bottom", 0),
                'fontname': word.get('fontname', ''),
                'size': word.get('size', 12),
                'is_bold': 'bold' in word.get('fontname', '').lower(),
                'width': word["x1"] - word["x0"]
            }
            lines_dict[y_key].append(word_info)
        
        # IMPROVED: Convert to structured lines with better column detection
        structured_lines = []
        for y_coord in sorted(lines_dict.keys()):
            line_words = sorted(lines_dict[y_coord], key=lambda w: w["x0"])
            
            # FIXED: Apply 20-character limit early and filter meaningful content
            line_texts = []
            x_positions = []
            for w in line_words:
                text = w["text"][:self.char_limit].strip() if w["text"] else ""
                if text and len(text) > 0:
                    line_texts.append(text)
                    x_positions.append(w['x0'])
            
            # IMPROVED: Require minimum meaningful content for table rows
            if len(line_texts) >= 2 and self._is_meaningful_line_content(line_texts):
                
                structured_lines.append({
                    'page': page_num,
                    'y_coord': y_coord,
                    'texts': line_texts,
                    'line_str': ' | '.join(line_texts),  # Better separator for debug
                    'x_positions': x_positions,
                    'word_count': len(line_texts),
                    'has_bold': any(w.get('is_bold', False) for w in line_words),
                    'avg_font_size': sum(w.get('size', 12) for w in line_words) / len(line_words),
                    'line_width': max(x_positions) - min(x_positions) if x_positions else 0
                })
        
        return {
            'lines': structured_lines,
            'word_groups': lines_dict
        }

    def _is_meaningful_line_content(self, texts: List[str]) -> bool:
        """Check if line contains meaningful content for tables."""
        combined = ' '.join(texts).strip()
        
        # Basic quality checks
        if len(combined) < 2:
            return False
        
        # Should have alphanumeric content
        if not re.search(r'[a-zA-Z0-9]', combined):
            return False
        
        # Skip lines that are just separators
        if re.match(r'^[\s\-_=|.]+$', combined):
            return False
        
        # Skip obvious non-table content
        if self.data_patterns['reject_lines'].search(combined):
            return False
        
        return True

    def _detect_and_format_tables(self, page_data: Dict, page_num: int) -> List[Dict]:
        """FIXED: Better table detection and column mapping."""
        lines = page_data['lines']
        
        if not lines:
            return []
        
        # IMPROVED: Find headers with better parameter matching
        header_candidates = self._find_headers_improved(lines)
        
        if self.debug:
            print(f"  Header candidates found: {len(header_candidates)}")
        
        tables = []
        for header_info in header_candidates:
            table = self._extract_table_with_fixed_columns(lines, header_info)
            if table:
                tables.append(table)
        
        return tables

    def _find_headers_improved(self, lines: List[Dict]) -> List[Dict]:
        """IMPROVED: Better header detection with column consistency."""
        header_candidates = []
        
        for i, line in enumerate(lines):
            line_texts = line['texts']
            line_str_lower = line['line_str'].lower()
            
            # Check for parameter matches
            found_params = {}
            param_score = 0
            
            for param_name, patterns in self.target_parameters.items():
                for pattern in patterns:
                    # IMPROVED: Better pattern matching
                    if any(pattern in text.lower() for text in line_texts) or pattern in line_str_lower:
                        found_params[param_name] = pattern
                        param_score += 1
                        break
            
            # IMPROVED: Require at least 2 parameters and reasonable column count
            if len(found_params) >= 2 and 3 <= len(line_texts) <= 10:
                
                # FIXED: Better column position tracking
                column_positions = self._analyze_column_positions(line)
                
                # ENHANCED: Header quality scoring
                header_score = self._calculate_header_quality(line, found_params)
                
                if self.debug:
                    print(f"  HEADER CANDIDATE: {line['line_str']}")
                    print(f"    Parameters: {list(found_params.keys())}")
                    print(f"    Score: {header_score:.2f}")
                    print(f"    Columns: {len(line_texts)}")
                
                header_candidates.append({
                    'line_info': line,
                    'line_index': i,
                    'found_params': found_params,
                    'param_score': param_score,
                    'header_score': header_score,
                    'column_positions': column_positions,
                    'column_count': len(line_texts)
                })
        
        # Sort by header quality score
        header_candidates.sort(key=lambda x: (x['param_score'], x['header_score']), reverse=True)
        
        return header_candidates[:3]  # Limit to top 3 tables

    def _calculate_header_quality(self, line: Dict, found_params: Dict) -> float:
        """Calculate header quality score."""
        score = 0.0
        
        # Parameter count score
        score += len(found_params) * 0.3
        
        # Font characteristics
        if line.get('has_bold', False):
            score += 0.2
        
        if line.get('avg_font_size', 12) > 12:
            score += 0.1
        
        # Line width (headers often span wider)
        if line.get('line_width', 0) > 300:
            score += 0.1
        
        # Word count (optimal header range)
        word_count = line.get('word_count', 0)
        if 4 <= word_count <= 8:
            score += 0.2
        elif 3 <= word_count <= 10:
            score += 0.1
        
        # Currency indicators
        line_str = line.get('line_str', '').lower()
        if any(curr in line_str for curr in ['â‚¬', 'eur', 'euro', 'price', 'preis']):
            score += 0.1
        
        return min(score, 1.0)

    def _analyze_column_positions(self, line: Dict) -> List[Dict]:
        """ENHANCED: Analyze column positions for better alignment."""
        x_positions = line.get('x_positions', [])
        texts = line.get('texts', [])
        
        columns = []
        for i, (x_pos, text) in enumerate(zip(x_positions, texts)):
            # IMPROVED: Better width estimation
            char_width = 7  # Average character width
            text_width = len(text) * char_width
            
            columns.append({
                'index': i,
                'x_start': x_pos,
                'x_end': x_pos + text_width,
                'x_center': x_pos + (text_width / 2),
                'text': text[:self.char_limit],  # Apply 20-char limit
                'width': text_width,
                'original_text': text  # Keep original for reference
            })
        
        return columns

    def _extract_table_with_fixed_columns(self, lines: List[Dict], header_info: Dict) -> Optional[Dict]:
        """FIXED: Extract table with proper column alignment."""
        header_line = header_info['line_info']
        header_index = header_info['line_index']
        column_positions = header_info['column_positions']
        expected_columns = header_info['column_count']
        
        if self.debug:
            print(f"\n--- EXTRACTING TABLE ---")
            print(f"Header: {header_line['line_str']}")
            print(f"Expected columns: {expected_columns}")
            print(f"Column positions: {[col['x_start'] for col in column_positions]}")
        
        # IMPROVED: Create column mapping
        column_mapping = self._create_column_mapping(header_line['texts'])
        
        # Extract table data
        table_rows = []
        header_row = [text[:self.char_limit] for text in header_line['texts']]  # Apply limit
        table_rows.append(header_row)
        
        data_rows_found = 0
        consecutive_empty = 0
        
        # FIXED: Better data row extraction with column alignment
        for line in lines[header_index + 1:]:
            # Check distance from header
            y_distance = abs(line['y_coord'] - header_line['y_coord'])
            if y_distance > 1000:  # Reasonable distance limit
                if self.debug:
                    print(f"  Stopped: Distance limit reached ({y_distance:.0f}px)")
                break
            
            if self._is_valid_data_row_fixed(line, column_positions, expected_columns):
                # FIXED: Align data to columns properly
                aligned_row = self._align_data_to_columns(line, column_positions, expected_columns)
                
                if aligned_row and any(cell.strip() for cell in aligned_row):  # Has meaningful content
                    table_rows.append(aligned_row)
                    data_rows_found += 1
                    consecutive_empty = 0
                    
                    if self.debug and data_rows_found <= 5:
                        print(f"  Row {data_rows_found}: {aligned_row}")
                else:
                    consecutive_empty += 1
                
                # Stop conditions
                if data_rows_found >= 100:  # Max rows limit
                    if self.debug:
                        print(f"  Stopped: Max rows limit reached")
                    break
                
                if consecutive_empty >= 5:  # Too many empty rows
                    if self.debug:
                        print(f"  Stopped: Too many consecutive empty rows")
                    break
            else:
                consecutive_empty += 1
                if consecutive_empty >= 8:  # Pattern break
                    if self.debug:
                        print(f"  Stopped: Pattern break detected")
                    break
        
        if self.debug:
            print(f"Table extraction complete: {len(table_rows)} total rows, {data_rows_found} data rows")
        
        # FIXED: Only return tables with meaningful data
        if len(table_rows) >= 2 and data_rows_found >= 1:
            # Calculate quality metrics
            quality_score = self._calculate_table_quality(table_rows, column_mapping)
            
            return {
                'page': header_line['page'],
                'headers': header_row,
                'data': table_rows,
                'table_data': table_rows[1:],  # Data without header
                'row_count': data_rows_found,
                'column_count': expected_columns,
                'column_mapping': column_mapping,
                'column_positions': column_positions,
                'found_params': header_info['found_params'],
                'header_score': header_info.get('header_score', 0),
                'quality_score': quality_score,
                'extraction_type': 'adaptive_fixed_20char'
            }
        else:
            if self.debug:
                print(f"  REJECTED: Insufficient data ({len(table_rows)} rows, {data_rows_found} data)")
        
        return None

    def _create_column_mapping(self, header_texts: List[str]) -> Dict[int, str]:
        """IMPROVED: Create mapping from column index to parameter type."""
        mapping = {}
        
        for col_idx, header_text in enumerate(header_texts):
            header_lower = header_text.lower()
            
            # Check each parameter type with fuzzy matching
            best_match = None
            for param_name, patterns in self.target_parameters.items():
                for pattern in patterns:
                    if pattern in header_lower or header_lower in pattern:
                        best_match = param_name
                        break
                if best_match:
                    break
            
            # Assign best match or unknown
            mapping[col_idx] = best_match if best_match else 'unknown'
        
        return mapping

    def _is_valid_data_row_fixed(self, line: Dict, column_positions: List[Dict], expected_columns: int) -> bool:
        """FIXED: Better validation for data rows."""
        texts = line.get('texts', [])
        line_str = line.get('line_str', '')
        x_positions = line.get('x_positions', [])
        
        # Basic checks
        if len(texts) < 1:
            return False
        
        # Reject obvious non-data
        reject_patterns = ['seite', 'page', 'blatt', 'confidential', 'proprietary', 'tel:', 'fax:']
        if any(pattern in line_str.lower() for pattern in reject_patterns):
            return False
        
        # IMPROVED: Check column alignment
        if len(x_positions) == 0:
            return False
        
        # FIXED: Better alignment checking
        alignment_matches = 0
        tolerance = 60  # Pixels tolerance for column alignment
        
        for line_x in x_positions[:expected_columns]:
            for col_info in column_positions:
                if abs(line_x - col_info['x_start']) <= tolerance:
                    alignment_matches += 1
                    break
        
        # IMPROVED: Require reasonable alignment
        alignment_ratio = alignment_matches / min(len(x_positions), expected_columns)
        
        # FIXED: Content validation
        has_meaningful_content = (
            any(re.search(r'\d', text) for text in texts) or  # Has numbers
            any(len(text.strip()) > 1 for text in texts) or   # Has meaningful text
            alignment_ratio >= 0.3  # Good alignment
        )
        
        return has_meaningful_content and alignment_ratio >= 0.2

    def _align_data_to_columns(self, line: Dict, column_positions: List[Dict], expected_columns: int) -> List[str]:
        """FIXED: Properly align data to column positions with 20-char limit."""
        texts = line.get('texts', [])
        x_positions = line.get('x_positions', [])
        
        if len(texts) == 0 or len(x_positions) == 0:
            return [""] * expected_columns
        
        # IMPROVED: Create aligned row
        aligned_row = [""] * expected_columns
        tolerance = 70  # Pixel tolerance for column matching
        
        # FIXED: Match each text to closest column
        for text, x_pos in zip(texts, x_positions):
            best_match_col = -1
            best_distance = float('inf')
            
            for col_info in column_positions:
                # IMPROVED: Use column center for better matching
                distance = abs(x_pos - col_info['x_start'])
                center_distance = abs(x_pos - col_info['x_center'])
                final_distance = min(distance, center_distance)
                
                if final_distance < tolerance and final_distance < best_distance:
                    best_distance = final_distance
                    best_match_col = col_info['index']
            
            # FIXED: Assign text to best matching column with 20-char limit
            if best_match_col >= 0 and best_match_col < expected_columns:
                cell_text = text[:self.char_limit].strip()
                
                # If cell already has content, append with separator
                if aligned_row[best_match_col]:
                    combined = f"{aligned_row[best_match_col]} {cell_text}"
                    aligned_row[best_match_col] = combined[:self.char_limit]
                else:
                    aligned_row[best_match_col] = cell_text
        
        return aligned_row

    def _calculate_table_quality(self, table_rows: List[List[str]], column_mapping: Dict[int, str]) -> float:
        """Calculate overall table quality score."""
        if len(table_rows) <= 1:
            return 0.0
        
        total_cells = 0
        filled_cells = 0
        param_columns = 0
        
        data_rows = table_rows[1:]  # Skip header
        
        # Calculate fill ratio
        for row in data_rows:
            for cell in row:
                total_cells += 1
                if cell and cell.strip():
                    filled_cells += 1
        
        fill_ratio = filled_cells / total_cells if total_cells > 0 else 0
        
        # Count parameter columns
        for param_type in column_mapping.values():
            if param_type != 'unknown':
                param_columns += 1
        
        param_ratio = param_columns / len(column_mapping) if column_mapping else 0
        
        # Combined quality score
        quality = (fill_ratio * 0.6 + param_ratio * 0.4)
        
        return quality
    
    
    def add_pos_column(self, table_rows):
        """Add 'pos' column if missing."""
        if not table_rows or 'pos' not in table_rows[0]:  # Assuming first row is headers
            table_rows[0].insert(0, 'pos')  # Add to headers
            for i, row in enumerate(table_rows[1:], start=1):
                row.insert(0, str(i))  # Add position to data rows
        return table_rows

    def render_table_to_image(self, table_rows, font_size=12):
        """Helper: Render table as image for LayoutLMv3 input."""
        if not table_rows:
            return None
        font = ImageFont.load_default()
        max_width = max(sum(font.getsize(str(cell))[0] for cell in row) + len(row) * 10 for row in table_rows) + 50
        max_height = len(table_rows) * (font_size + 5) + 50
        img = Image.new('RGB', (max_width, max_height), color='white')
        draw = ImageDraw.Draw(img)
        y = 10
        for row in table_rows:
            x = 10
            for cell in row:
                draw.text((x, y), str(cell), fill='black', font=font)
                x += font.getsize(str(cell))[0] + 10
            y += font_size + 5
        return img

    def filter_tables_with_layoutlmv3(self, candidate_tables):
        refined_tables = []
        if self.debug:
            print(f"Entering filter: {len(candidate_tables)} candidate tables found by heuristics")
        
        for table in candidate_tables:
            table_rows = [table.get('headers', [])] + table.get('table_data', [])
            if not table_rows:
                if self.debug:
                    print("Skipped: Empty table rows")
                continue
            
            table_rows = self.add_pos_column(table_rows)
            table_img = self.render_table_to_image(table_rows)
            if table_img is None:
                if self.debug:
                    print("Skipped: Failed to render image")
                continue
            
            try:
                inputs = self.processor(table_img, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                logits = outputs.logits
                if logits.numel() == 0:
                    if self.debug:
                        print("Skipped: No logits from model (empty input)")
                    continue
                
                pred_classes = logits.argmax(-1)[0]
                pred_class = torch.mode(pred_classes).mode.item() if len(pred_classes) > 0 else -1
                confidence = torch.softmax(logits, dim=-1)[0, :, pred_class].mean().item() if len(pred_classes) > 0 else 0.0
                
                if self.debug:
                    print(f"Table Debug: Predicted Class = {pred_class}, Confidence = {confidence:.2f}")
                    print(f"  Expected Class: {self.TARGET_CLASS_INDEX}, Threshold: {self.CONFIDENCE_THRESHOLD}")
                    if pred_class == self.TARGET_CLASS_INDEX and confidence >= self.CONFIDENCE_THRESHOLD:
                        print("  Accepted!")
                    else:
                        print("  Rejected! (Class mismatch or low confidence)")
                
                if pred_class == self.TARGET_CLASS_INDEX and confidence >= self.CONFIDENCE_THRESHOLD:
                    refined_table = table.copy()
                    refined_table['headers'] = table_rows[0]
                    refined_table['table_data'] = table_rows[1:]
                    refined_tables.append(refined_table)
            
            except Exception as e:
                if self.debug:
                    print(f"Error during inference: {e}")
                continue
        
        if self.debug:
            print(f"Exiting filter: {len(refined_tables)} tables kept")
        return refined_tables



# ==================================================================================
# ENHANCED DISPLAY FUNCTIONS WITH ADAPTIVE LEARNING INFO
# ==================================================================================

def display_adaptive_tables(tables: List[Dict], show_learning_info: bool = True, max_rows: int = 20):
    """Display tables with adaptive learning information."""
    
    if not tables:
        print("âŒ No tables found!")
        return
    
    for table_num, table in enumerate(tables, 1):
        print(f"\n{'='*100}")
        print(f"ðŸ“Š ADAPTIVE TABLE {table_num} - LEARNING ENABLED (20-CHAR LIMIT)")
        print(f"{'='*100}")
        
        headers = table.get('headers', [])
        data_rows = table.get('table_data', [])
        column_mapping = table.get('column_mapping', {})
        found_params = table.get('found_params', {})
        quality_score = table.get('quality_score', 0)
        extraction_method = table.get('extraction_method', 'unknown')
        document_hash = table.get('document_hash', 'unknown')
        
        print(f"ðŸ“„ Page: {table.get('page', 'Unknown')}")
        print(f"ðŸ“Š Size: {len(data_rows)} rows Ã— {len(headers)} columns")
        print(f"ðŸŽ¯ Found Parameters: {list(found_params.keys())}")
        print(f"ðŸ“ˆ Quality Score: {quality_score:.2f}")
        print(f"ðŸ”§ Extraction Method: {extraction_method}")
        
        if show_learning_info:
            print(f"ðŸ§  Document Hash: {document_hash[:12]}...")
        
        # Show column mapping
        print(f"\nðŸ”— Column Mapping (20-char limit applied):")
        for col_idx, param_type in column_mapping.items():
            header_text = headers[col_idx] if col_idx < len(headers) else 'N/A'
            param_display = param_type.replace('_', '-').title()
            learning_indicator = "ðŸ§ " if param_type != 'unknown' else "â“"
            print(f"  {learning_indicator} Column {col_idx}: '{header_text}' â†’ {param_display}")
        
        # Display table with proper alignment
        print(f"\nðŸ“‹ TABLE DATA (Each cell limited to 20 characters):")
        print("-" * 100)
        
        # Header row
        header_display = []
        for i, header in enumerate(headers):
            param_type = column_mapping.get(i, 'unknown')
            indicators = {
                'artikel_nr': '[Art#]',
                'beschreibung': '[Desc]',
                'einzelpreis': '[Unitâ‚¬]',
                'gesamtpreis': '[Totâ‚¬]',
                'menge': '[Qty]',
                'unknown': '[?]'
            }
            indicator = indicators.get(param_type, '[?]')
            
            display_text = f"{header[:15]} {indicator}"
            header_display.append(f"{display_text:22}"[:22])
        
        print(f"{'|'.join(header_display)}")
        print("-" * 100)
        
        # Data rows with enhanced formatting
        display_count = min(max_rows, len(data_rows))
        for i, row in enumerate(data_rows[:display_count], 1):
            row_display = []
            for j, cell in enumerate(row):
                param_type = column_mapping.get(j, 'unknown')
                
                cell_text = str(cell)[:20].strip() if cell else ""
                
                if param_type in ['einzelpreis', 'gesamtpreis'] and re.search(r'\d', cell_text):
                    if not cell_text.startswith('â‚¬') and re.search(r'\d', cell_text):
                        formatted_cell = f"â‚¬{cell_text}"
                    else:
                        formatted_cell = cell_text
                elif param_type == 'menge' and re.search(r'^\d+$', cell_text):
                    formatted_cell = f"{cell_text}x"
                elif param_type == 'artikel_nr':
                    formatted_cell = cell_text.upper() if cell_text else ""
                else:
                    formatted_cell = cell_text
                
                row_display.append(f"{formatted_cell:22}"[:22])
            
            print(f"{'|'.join(row_display)}")
        
        if len(data_rows) > display_count:
            print(f"... and {len(data_rows) - display_count} more rows")
        
        print("-" * 100)
        
        # Show comprehensive statistics
        print(f"\nðŸ“ˆ TABLE STATISTICS:")
        stats = calculate_comprehensive_stats(table)
        for key, value in stats.items():
            print(f"  {key}: {value}")


def calculate_comprehensive_stats(table: Dict) -> Dict[str, str]:
    """Calculate comprehensive statistics for the adaptive table."""
    data_rows = table.get('table_data', [])
    column_mapping = table.get('column_mapping', {})
    quality_score = table.get('quality_score', 0)
    
    stats = {}
    stats['Total Rows'] = str(len(data_rows))
    stats['Total Columns'] = str(table.get('column_count', 0))
    stats['Quality Score'] = f"{quality_score:.2f}"
    stats['Learning Enabled'] = "Yes âœ…"
    
    # Count non-empty cells
    non_empty_cells = 0
    total_cells = 0
    
    for row in data_rows:
        for cell in row:
            total_cells += 1
            if cell and cell.strip():
                non_empty_cells += 1
    
    if total_cells > 0:
        stats['Data Completeness'] = f"{(non_empty_cells/total_cells)*100:.1f}%"
    
    # Parameter coverage
    target_params = {'artikel_nr', 'beschreibung', 'einzelpreis', 'gesamtpreis', 'menge'}
    found_params = set(column_mapping.values()) - {'unknown'}
    coverage = len(found_params.intersection(target_params)) / len(target_params)
    stats['Parameter Coverage'] = f"{coverage*100:.1f}%"
    
    # Calculate totals for numeric columns
    total_einzelpreis = 0
    total_gesamtpreis = 0
    total_menge = 0
    artikel_count = 0
    
    for row in data_rows:
        for col_idx, cell in enumerate(row):
            param_type = column_mapping.get(col_idx, 'unknown')
            
            if param_type == 'artikel_nr' and cell.strip():
                artikel_count += 1
            elif param_type == 'einzelpreis' and cell.strip():
                try:
                    price = float(re.sub(r'[^\d.,]', '', cell).replace(',', '.'))
                    total_einzelpreis += price
                except:
                    pass
            elif param_type == 'gesamtpreis' and cell.strip():
                try:
                    price = float(re.sub(r'[^\d.,]', '', cell).replace(',', '.'))
                    total_gesamtpreis += price
                except:
                    pass
            elif param_type == 'menge' and cell.strip():
                try:
                    qty = float(re.sub(r'[^\d.,]', '', cell).replace(',', '.'))
                    total_menge += qty
                except:
                    pass
    
    if artikel_count > 0:
        stats['Articles Found'] = str(artikel_count)
    if total_einzelpreis > 0:
        stats['Total Unit Prices'] = f"â‚¬{total_einzelpreis:.2f}"
    if total_gesamtpreis > 0:
        stats['Total Line Prices'] = f"â‚¬{total_gesamtpreis:.2f}"
    if total_menge > 0:
        stats['Total Quantity'] = f"{total_menge:.0f}"
    
    return stats


# ==================================================================================
# MAIN EXTRACTION FUNCTIONS WITH ADAPTIVE LEARNING
# ==================================================================================

def extract_tables_with_adaptive_learning(file_path: str, debug: bool = False, learning_db_path: str = "table_learning.db") -> List[Dict]:
    """Extract tables with adaptive learning enabled."""
    extractor = AdaptiveFixedTableExtractor(debug=debug, learning_db_path=learning_db_path)
    return extractor.extract_tables_with_learning(file_path)

def save_table_corrections(file_path: str, corrections: Dict[str, Any], learning_db_path: str = "table_learning.db"):
    """Save user corrections for adaptive learning."""
    with open(file_path, 'rb') as f:
        document_hash = hashlib.sha256(f.read()).hexdigest()[:32]
    
    extractor = AdaptiveFixedTableExtractor(debug=False, learning_db_path=learning_db_path)
    extractor.save_user_corrections(document_hash, corrections)

def get_learning_statistics(learning_db_path: str = "table_learning.db") -> Dict[str, Any]:
    """Get learning statistics."""
    extractor = AdaptiveFixedTableExtractor(debug=False, learning_db_path=learning_db_path)
    return extractor.get_learning_statistics()

def extract_and_display_adaptive(file_path: str, debug: bool = False, max_rows: int = 20, learning_db_path: str = "table_learning.db"):
    """Extract and display tables with adaptive learning features."""
    print(f"ðŸ§  ADAPTIVE FIXED TABLE EXTRACTION WITH LEARNING")
    print(f"âœ… Improved column alignment & data grouping")
    print(f"âœ… 20-character limit per cell")
    print(f"âœ… Adaptive learning from user corrections")
    print(f"âœ… Pattern recognition and improvement")
    print(f"File: {os.path.basename(file_path)}")
    print("=" * 80)
    
    # Extract with adaptive learning
    tables = extract_tables_with_adaptive_learning(file_path, debug=debug, learning_db_path=learning_db_path)
    
    if tables:
        print(f"\nâœ… SUCCESS! Found {len(tables)} table(s) with adaptive learning enabled")
        
        # Display with adaptive learning info
        display_adaptive_tables(tables, show_learning_info=True, max_rows=max_rows)
        
        # Show learning statistics
        extractor = AdaptiveFixedTableExtractor(debug=debug, learning_db_path=learning_db_path)
        learning_stats = extractor.get_learning_statistics()
        
        print(f"\nðŸ§  ADAPTIVE LEARNING SUMMARY:")
        print(f"  ðŸ“Š Total corrections saved: {learning_stats['total_corrections']}")
        print(f"  ðŸŽ¯ Learned patterns: {learning_stats['learned_patterns']}")
        print(f"  ðŸ“ Documents with corrections: {learning_stats['documents_with_corrections']}")
        print(f"  ðŸ“ˆ Learning rate: {learning_stats['learning_rate']:.1f}%")
        print(f"  ðŸ’¾ Database: {learning_db_path}")
        
        # Show parameter coverage across all tables
        all_params = set()
        for table in tables:
            column_mapping = table.get('column_mapping', {})
            all_params.update(column_mapping.values())
        
        all_params.discard('unknown')
        target_params = {'artikel_nr', 'beschreibung', 'einzelpreis', 'gesamtpreis', 'menge'}
        overall_coverage = len(all_params.intersection(target_params)) / len(target_params)
        
        print(f"\nðŸŽ¯ Parameters found across all tables:")
        param_names = {
            'artikel_nr': 'Artikel-Nr/Position',
            'beschreibung': 'Beschreibung/Description', 
            'einzelpreis': 'Einzelpreis/Unit Price',
            'gesamtpreis': 'Gesamtpreis/Total Price',
            'menge': 'Menge/Quantity'
        }
        
        for param_key, param_name in param_names.items():
            status = "âœ…" if param_key in all_params else "âŒ"
            print(f"  {status} {param_name}")
        
        print(f"\nðŸ“ˆ Overall parameter coverage: {overall_coverage:.1%}")
        
    else:
        print("âŒ No tables found with the adaptive extraction method")
        print("ðŸ’¡ Try with a different PDF containing clear business table structures")
    
    return tables

# ==================================================================================
# INTERACTIVE MODE WITH ADAPTIVE LEARNING FEATURES
# ==================================================================================

def interactive_adaptive_mode():
    """Interactive mode for adaptive table extraction with learning features."""
    print("=" * 80)
    print("ADAPTIVE FIXED TABLE EXTRACTION WITH LEARNING")
    print("Features: Column alignment, 20-char limit, adaptive learning, pattern recognition")
    print("=" * 80)
    
    learning_db_path = "table_learning.db"
    
    while True:
        print("\nOptions:")
        print("1. Extract with adaptive learning (show 15 rows)")
        print("2. Extract with adaptive learning (show 30 rows)")
        print("3. Extract with adaptive learning (show ALL rows)")
        print("4. Extract with debug output")
        print("5. Show learning statistics")
        print("6. Clear learning data")
        print("7. Test correction saving")
        print("8. Exit")
        
        choice = input("\nChoice (1-8): ").strip()
        
        if choice in ["1", "2", "3", "4"]:
            pdf_path = input("\nEnter PDF path: ").strip().strip('"').strip("'")
            
            if not os.path.exists(pdf_path):
                print(f"âŒ File not found!")
                continue
            
            try:
                if choice == "1":
                    extract_and_display_adaptive(pdf_path, debug=False, max_rows=15, learning_db_path=learning_db_path)
                elif choice == "2":
                    extract_and_display_adaptive(pdf_path, debug=False, max_rows=30, learning_db_path=learning_db_path)
                elif choice == "3":
                    extract_and_display_adaptive(pdf_path, debug=False, max_rows=1000, learning_db_path=learning_db_path)
                elif choice == "4":
                    extract_and_display_adaptive(pdf_path, debug=True, max_rows=20, learning_db_path=learning_db_path)
                
            except Exception as e:
                print(f"âŒ Error processing PDF: {str(e)}")
                import traceback
                traceback.print_exc()
        
        elif choice == "5":
            # Show learning statistics
            try:
                stats = get_learning_statistics(learning_db_path)
                print(f"\nðŸ“Š LEARNING STATISTICS:")
                print(f"  Total corrections: {stats['total_corrections']}")
                print(f"  Learned patterns: {stats['learned_patterns']}")
                print(f"  Documents with corrections: {stats['documents_with_corrections']}")
                print(f"  Total documents processed: {stats['total_documents_processed']}")
                print(f"  Average corrections per document: {stats['average_corrections_per_document']:.1f}")
                print(f"  Learning rate: {stats['learning_rate']:.1f}%")
                
                if stats['correction_types']:
                    print(f"\n  Correction types:")
                    for corr_type, count in stats['correction_types']:
                        print(f"    {corr_type}: {count}")
                
                if stats['pattern_statistics']:
                    print(f"\n  Pattern statistics:")
                    for param_type, avg_conf, pattern_count in stats['pattern_statistics']:
                        print(f"    {param_type}: {pattern_count} patterns (avg confidence: {avg_conf:.2f})")
                        
            except Exception as e:
                print(f"âŒ Error getting statistics: {e}")
        
        elif choice == "6":
            # Clear learning data
            confirm = input("âš ï¸ Clear all learning data? (yes/no): ").strip().lower()
            if confirm == 'yes':
                try:
                    extractor = AdaptiveFixedTableExtractor(debug=False, learning_db_path=learning_db_path)
                    extractor.clear_learning_data()
                    print("âœ… Learning data cleared!")
                except Exception as e:
                    print(f"âŒ Error clearing data: {e}")
        
        elif choice == "7":
            # Test correction saving
            pdf_path = input("\nEnter PDF path for correction test: ").strip().strip('"').strip("'")
            if os.path.exists(pdf_path):
                try:
                    # Example correction data
                    test_corrections = {
                        "0": {  # Table 0
                            "header": {
                                "0": {"original": "pos", "corrected": "Position"},
                                "1": {"original": "desc", "corrected": "Beschreibung"}
                            },
                            "cell": {
                                "0_0": {"original": "1", "corrected": "1"},
                                "0_1": {"original": "test", "corrected": "Test Item"}
                            }
                        }
                    }
                    
                    save_table_corrections(pdf_path, test_corrections, learning_db_path)
                    print("âœ… Test corrections saved successfully!")
                    
                except Exception as e:
                    print(f"âŒ Error saving test corrections: {e}")
        
        elif choice == "8":
            print("ðŸ‘‹ Goodbye!")
            break
        else:
            print("âŒ Invalid choice!")


# ==================================================================================
# BACKWARDS COMPATIBILITY
# ==================================================================================

# Keep original class name for backwards compatibility
FixedTableExtractor = AdaptiveFixedTableExtractor

# Keep original functions for backwards compatibility  
def extract_fixed_tables(file_path: str, debug: bool = False) -> List[Dict]:
    """Extract tables with fixed column alignment and 20-char limit (backwards compatible)."""
    return extract_tables_with_adaptive_learning(file_path, debug=debug)

def display_fixed_tables(tables: List[Dict], show_alignment: bool = True, max_rows: int = 20):
    """Display tables with fixed formatting (backwards compatible)."""
    display_adaptive_tables(tables, show_learning_info=show_alignment, max_rows=max_rows)

def extract_and_display_fixed(file_path: str, debug: bool = False, max_rows: int = 20):
    """Extract and display tables with all fixes applied (backwards compatible)."""
    return extract_and_display_adaptive(file_path, debug=debug, max_rows=max_rows)

def interactive_fixed_mode():
    """Interactive mode for fixed table extraction (backwards compatible)."""
    return interactive_adaptive_mode()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        pdf_file = sys.argv[1]
        debug_mode = "--debug" in sys.argv
        show_all = "--all" in sys.argv
        learning_mode = "--learning" in sys.argv or "--adaptive" in sys.argv
        
        if os.path.exists(pdf_file):
            max_display = 1000 if show_all else 20
            
            if learning_mode:
                extract_and_display_adaptive(pdf_file, debug=debug_mode, max_rows=max_display)
            else:
                # Backwards compatible mode
                extract_and_display_fixed(pdf_file, debug=debug_mode, max_rows=max_display)
        else:
            print(f"âŒ File not found: {pdf_file}")
    else:
        # Interactive mode
        print("ðŸš€ Starting Adaptive Table Extraction...")
        print("Use --learning flag for adaptive learning mode")
        interactive_adaptive_mode()