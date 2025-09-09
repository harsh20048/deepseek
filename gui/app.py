"""
Updated Adaptive PDF Table Extraction Application
Refined version with consolidated imports, improved error handling, and cleaner structure.
"""

import sys
import copy
import os
import tempfile
import sqlite3
import json
import time
import hashlib
import logging
import subprocess
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to Python path for proper imports - MUST BE FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Adaptive PDF Data Extraction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adaptive table extraction imports with comprehensive fallbacks
# Adaptive table extraction imports with comprehensive fallbacks
try:
    from extraction.feild_extraction import AdaptiveFixedTableExtractor, extract_fields_with_regex_fallback
    extract_learning_available = True
    print("âœ… Field extraction imports successful")
except ImportError as e:
    extract_learning_available = False
    print(f"âŒ Field extraction import error: {e}")
    
    # Create fallback class
    class AdaptiveFixedTableExtractor:
        def __init__(self, *args, **kwargs):
            pass

try:
    from extraction.table_extraction import (
        extract_tables_with_adaptive_learning,
        save_table_corrections,
        get_learning_statistics,
        AdaptiveFixedTableExtractor as AdaptiveTableExtractor,
        extract_and_display_adaptive
    )
    ADAPTIVE_AVAILABLE = True
    print("âœ… Table extraction imports successful")
except ImportError as e:
    ADAPTIVE_AVAILABLE = False
    print(f"âŒ Table extraction import error: {e}")
    # (keep your fallback AdaptiveTableExtractor here)

    # Complete fallback implementation
    class AdaptiveTableExtractor:
        def __init__(self, *args, **kwargs):
            self.debug = kwargs.get('debug', False)

        def extract_tables_with_learning(self, *args, **kwargs):
            return []

        def bulk_save_corrections(self, *args, **kwargs):
            return 0

        def get_learning_statistics(self, *args, **kwargs):
            return {'templates_learned': 0, 'header_corrections': 0}

    def extract_tables_with_learning(*args, **kwargs):
        return []

    def extract_tables_with_adaptive_learning(*args, **kwargs):
        return []

    def save_table_corrections(*args, **kwargs):
        return True

    def get_learning_statistics(*args, **kwargs):
        return {'templates_learned': 0, 'header_corrections': 0}

    def extract_and_display_adaptive(*args, **kwargs):
        return []

# Field extraction imports
try:
    from extraction.feild_extraction import (
        PDFFieldExtractor,
        extract_fields_from_pdf,
        process_pdf_documents
    )
    FIELD_EXTRACTION_AVAILABLE = True
    print("âœ… Field extraction imports successful")
except ImportError as e:
    FIELD_EXTRACTION_AVAILABLE = False
    print(f"âŒ Field extraction import error: {e}")
    # (keep your fallback PDFFieldExtractor here)

    # Fallback PDFFieldExtractor
    class PDFFieldExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract_fields(self, text, **kwargs):
            return {'date': '', 'angebot': '', 'company_name': '', 'sender_address': ''}

    def extract_fields_from_pdf(*args, **kwargs):
        return {'date': '', 'angebot': '', 'company_name': '', 'sender_address': ''}

    def process_pdf_documents(*args, **kwargs):
        return []

# Text extraction imports
try:
    from extraction.text_extraction import extract_text_from_pdf, detect_language
    print("âœ… Text extraction imports successful")
except ImportError as e:
    print(f"âŒ Text extraction import error: {e}")
    # (keep your pdfplumber fallback here)
except ImportError:
    pdfplumber = None

    def extract_text_from_pdf(file_path):
        return "Sample text", False

    def detect_language(text):
        return "en"

    class PDFFieldExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract_fields(self, text, **kwargs):
            return {'date': '', 'angebot': '', 'company_name': '', 'sender_address': ''}

# Learning system imports with comprehensive fallbacks
# âœ… NEW - Correct and comprehensive

# Project root path already added at the top of the file

# Global fallback functions for text extraction (in case imports fail)
def extract_text_from_pdf_fallback(file_path):
    """Fallback text extraction function."""
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text, False
    except:
        return "Sample text", False

def detect_language_fallback(text):
    """Fallback language detection function."""
    return "en"

try:
    from learning.database import LearningDatabase  # âœ… Correct filename
    from learning.pattern_learner import PatternLearner
    print("âœ… Learning system imports successful")
    LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"âŒ Learning system import failed: {e}")
    LEARNING_AVAILABLE = False

    # Fallback classes if needed
    class LearningDatabase:
        def __init__(self, *args, **kwargs):
            pass

        def get_table_correction_stats(self):
            return {'total_table_corrections': 0}

    class PatternLearner:
        def __init__(self, *args, **kwargs):
            pass

    print("Learning system modules imported successfully")
except ImportError:
    print("Â Learning system modules missing, using comprehensive fallbacks")

    class LearningDatabase:
        def __init__(self, db_path=None):
            self.db_path = db_path
            print(f"Â Fallback LearningDatabase initialized with {db_path}")

        def get_table_correction_stats(self):
            return {
                'total_table_corrections': 0,
                'average_confidence_improvement': 0.0,
                'documents_with_table_corrections': 0,
                'learned_pattern_count': 0
            }

        def save_document_metadata(self, doc_hash, filename, file_size):
            return True

        def get_corrections_for_document(self, doc_hash):
            return {}

        def save_correction(self, *args, **kwargs):
            return True

        def get_learned_table_patterns(self):
            return []

    class PatternLearner:
        def __init__(self, db_path=None):
            self.db_path = db_path
            print(f"Â Fallback PatternLearner initialized with {db_path}")

        def learn_from_feedback(self, *args, **kwargs):
            return True


def plot_chart(chart_type, data, title="", **kwargs):
    """Create charts using Streamlit native charts with enhanced options."""
    try:
        if chart_type == "bar":
            if 'x' in kwargs and 'y' in kwargs:
                chart_data = data.set_index(kwargs['x'])[[kwargs['y']]]
            else:
                chart_data = data
            st.bar_chart(chart_data)
        elif chart_type == "line":
            if 'x' in kwargs and 'y' in kwargs:
                chart_data = data.set_index(kwargs['x'])[[kwargs['y']]]
            else:
                chart_data = data
            st.line_chart(chart_data)
        else:
            st.dataframe(data)
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        st.dataframe(data)


class PDFExtractorApp:
    """Enhanced PDF Extraction Application with Adaptive Learning."""

    def __init__(self):
        """Initialize application with robust error handling and comprehensive setup."""
        from pathlib import Path
        import os
        import sys

        print("ðŸš€ Initializing PDF Extractor Application...")

        # Initialize session state first
        try:
            self.setup_session_state()
            print("âœ… Session state initialized")
        except Exception as e:
            print(f"âš ï¸ Warning: Session state initialization failed: {e}")

        # Set up base path with absolute resolution
        try:
            self.base_path = Path(__file__).parent.absolute()
            print(f"ðŸ“ Project root: {self.base_path}")
        except Exception as e:
            print(f"âš ï¸ Warning: Could not resolve project path: {e}")
            self.base_path = Path.cwd()
            print(f"ðŸ“ Using current directory: {self.base_path}")

        # Set up database paths consistently
        try:
            self.db_paths = {
                'adaptive_db': self.base_path / "adaptive_table_learning.db",
                'learning_db': self.base_path.parent / "learning" / "l2.db"
            }

            print(f"ðŸ—ƒï¸ Database paths configured:")
            for db_name, db_path in self.db_paths.items():
                print(f"   - {db_name}: {db_path}")

        except Exception as e:
            print(f"âŒ Error setting up database paths: {e}")
            self.db_paths = {}

        # Ensure directories exist
        try:
            for db_name, db_path in self.db_paths.items():
                db_path.parent.mkdir(parents=True, exist_ok=True)
                if db_path.parent.exists():
                    print(
                        f"âœ… Directory ensured for {db_name}: {db_path.parent}")
                else:
                    print(
                        f"âŒ Failed to create directory for {db_name}: {db_path.parent}")

        except Exception as e:
            print(f"âŒ Error creating directories: {e}")

        # Initialize Pattern Learner early
        try:
            from learning.pattern_learner import PatternLearner
            learning_db_path = self.db_paths.get('learning_db', 'learning.db')
            self.pattern_learner = PatternLearner(str(learning_db_path))
            print("âœ… Pattern learner initialized successfully")
        except ImportError as ie:
            print(f"âŒ Pattern learner import failed: {ie}")
            print("   Make sure learning/pattern_learner.py exists")
            self.pattern_learner = None
        except Exception as e:
            print(f"âŒ Pattern learner initialization error: {e}")
            self.pattern_learner = None

        # Initialize components with individual error handling
        print("ðŸ”§ Initializing system components...")

        # 1. Initialize adaptive extractor
        try:
            self._init_adaptive_extractor()
            print("âœ… Adaptive extractor initialized")
        except Exception as e:
            print(f"âš ï¸ Adaptive extractor initialization failed: {e}")

        # 2. Initialize learning systems
        try:
            self._init_learning_systems()
            print("âœ… Learning systems initialized")
        except Exception as e:
            print(f"âš ï¸ Learning systems initialization failed: {e}")

        # 3. Initialize additional components
        try:
            self._init_additional_components()
            print("âœ… Additional components initialized")
        except Exception as e:
            print(f"âš ï¸ Additional components initialization failed: {e}")

        # 4. Verify system health
        try:
            self._verify_system_health()
            print("âœ… System health verification completed")
        except Exception as e:
            print(f"âš ï¸ System health verification failed: {e}")

        # Print initialization summary
        print("\nðŸ“Š Initialization Summary:")
        print(f"   - Base path: {self.base_path}")
        print(
            f"   - Learning DB: {self.db_paths.get('learning_db', 'Not set')}")
        print(
            f"   - Pattern learner: {'âœ… Active' if self.pattern_learner else 'âŒ Failed'}")
        print(
            f"   - Learning DB: {'âœ… Active' if hasattr(self, 'learning_db') and self.learning_db else 'âŒ Failed'}")

        print("ðŸŽ‰ Application initialization complete!")

        # Optional: Test database connection
        if hasattr(self, 'learning_db') and self.learning_db:
            try:
                stats = self.learning_db.get_table_correction_stats()
                corrections_count = stats.get('total_table_corrections', 0)
                print(
                    f"ðŸ“Š Database connection verified: {corrections_count} corrections stored")
            except Exception as e:
                print(f"âš ï¸ Database connection test failed: {e}")

    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract tables from PDF using multiple methods with fallbacks.
        This is the actual working table extraction method.
        """
        tables = []
        
        try:
            # Method 1: Try to use the table_extraction.py module directly
            try:
                # Import the actual working table extraction
                import sys
                extraction_path = self.base_path.parent / "extraction"
                if str(extraction_path) not in sys.path:
                    sys.path.insert(0, str(extraction_path))
                
                from table_extraction import AdaptiveFixedTableExtractor
                
                st.write("🔍 DEBUG: Using AdaptiveFixedTableExtractor...")
                extractor = AdaptiveFixedTableExtractor(debug=True)
                tables = extractor.extract_tables_with_learning(pdf_path)
                
                if tables:
                    st.write(f"✅ DEBUG: AdaptiveFixedTableExtractor found {len(tables)} tables")
                    return tables
                else:
                    st.write("⚠️ DEBUG: AdaptiveFixedTableExtractor found no tables")
                    
            except Exception as e:
                st.write(f"⚠️ DEBUG: AdaptiveFixedTableExtractor failed: {e}")
            
            # Method 2: Try pdfplumber direct extraction
            try:
                import pdfplumber
                st.write("🔍 DEBUG: Using pdfplumber extraction...")
                
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_tables = page.extract_tables()
                        
                        for table_idx, table in enumerate(page_tables):
                            if table and len(table) > 1:
                                # Convert to our format
                                headers = table[0] if table[0] and any(table[0]) else None
                                rows = [row for row in table[1:] if row and any(cell and str(cell).strip() for cell in row)]
                                
                                if headers and rows:
                                    formatted_table = {
                                        'headers': [str(h).strip() if h else f"Column_{i+1}" for i, h in enumerate(headers)],
                                        'rows': [[str(cell).strip() if cell else "" for cell in row] for row in rows],
                                        'title': f"Table from Page {page_num}",
                                        'page': page_num,
                                        'table_index': table_idx + 1
                                    }
                                    tables.append(formatted_table)
                
                if tables:
                    st.write(f"✅ DEBUG: pdfplumber found {len(tables)} tables")
                    return tables
                else:
                    st.write("⚠️ DEBUG: pdfplumber found no tables")
                    
            except Exception as e:
                st.write(f"⚠️ DEBUG: pdfplumber failed: {e}")
            
            # Method 3: Use test.py fallback (as last resort)
            try:
                st.write("🔍 DEBUG: Using test.py fallback for tables...")
                result = self.run_test_py_fallback(pdf_path, 'tables')
                
                if result['success'] and result['data']:
                    if isinstance(result['data'], dict) and 'tables' in result['data']:
                        tables = result['data']['tables']
                    elif isinstance(result['data'], list):
                        tables = result['data']
                    
                    if tables:
                        st.write(f"✅ DEBUG: test.py fallback found {len(tables)} tables")
                        return tables
                    else:
                        st.write("⚠️ DEBUG: test.py fallback found no tables")
                else:
                    st.write(f"⚠️ DEBUG: test.py fallback failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.write(f"⚠️ DEBUG: test.py fallback exception: {e}")
        
        except Exception as e:
            st.error(f"❌ Table extraction completely failed: {e}")
        
        return tables

    def run(self):
        """Run the complete Streamlit application."""

        self.render_header()
        self.render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Adaptive Table Extraction",
            " Performance Dashboard",
            " Learning Progress",
            " Training Data",
            " Settings"
        ])

        with tab1:
            self.render_adaptive_table_extraction()

        with tab2:
            self.render_performance_dashboard()

        with tab3:
            self.render_learning_progress()

        with tab4:
            self.render_training_data()

        with tab5:
            self.render_settings()

    def _init_adaptive_extractor(self):
        """Initialize adaptive table extractor with fallback."""
        self.adaptive_extractor = None

        if ADAPTIVE_AVAILABLE:
            try:
                self.adaptive_extractor = AdaptiveTableExtractor(
                    debug=True,
                    learning_db_path=str(self.db_paths['adaptive_db'])
                )

                # Test the extractor
                stats = self.adaptive_extractor.get_learning_statistics()
                patterns = stats.get('patterns_learned', 0)
                templates = stats.get('templates_learned', 0)

                print(
                    f"Adaptive extractor initialized: {patterns} patterns, {templates} templates")

            except Exception as e:
                print(f"ÂÅ’ Error initializing adaptive extractor: {e}")
                self.adaptive_extractor = AdaptiveTableExtractor()  # Use fallback
        else:
            self.adaptive_extractor = AdaptiveTableExtractor()  # Use fallback
            print("Â Using fallback adaptive extractor")

    def _init_learning_systems(self):
        """Initialize learning database with comprehensive error handling."""

        learning_db_path = self.db_paths['learning_db']
        print(f"ðŸ—ƒï¸ Initializing learning database: {learning_db_path}")

        try:
            # Import after path setup to avoid circular imports
            from learning.database import LearningDatabase  # âœ… Correct

            # Initialize database with explicit path
            self.learning_db = LearningDatabase(db_path=str(learning_db_path))

            # Test the connection immediately
            try:
                stats = self.learning_db.get_table_correction_stats()
                corrections_count = stats.get('total_table_corrections', 0)
                print(
                    f"âœ… Learning database connected: {corrections_count} corrections stored")

                # Test a simple correction save/load
                test_hash = "test_connection_hash"
                try:
                    # Try to save a test correction
                    self.learning_db.save_correction(
                        document_hash=test_hash,
                        pdf_name="connection_test.pdf",
                        field_name="test_field",
                        corrected_value="test_value"
                    )

                    # Try to retrieve it
                    test_corrections = self.learning_db.get_corrections_for_document(
                        test_hash)
                    if test_corrections.get('test_field') == 'test_value':
                        print("âœ… Database read/write test successful")

                        # Clean up test data
                        import sqlite3
                        with sqlite3.connect(str(learning_db_path)) as conn:
                            conn.execute(
                                "DELETE FROM corrections WHERE document_hash = ?", (test_hash,))
                            conn.commit()
                    else:
                        print("âš ï¸ Database write test failed")

                except Exception as test_error:
                    print(f"âš ï¸ Database test failed: {test_error}")

            except Exception as stats_error:
                print(f"âš ï¸ Could not get database stats: {stats_error}")

        except Exception as e:
            print(f"âŒ Learning database initialization failed: {e}")
            print(f"âŒ Error details: {type(e).__name__}: {str(e)}")

            # Try to create a minimal database
            try:
                import sqlite3
                with sqlite3.connect(str(learning_db_path)) as conn:
                    # Create minimal tables
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS corrections (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            document_hash TEXT NOT NULL,
                            pdf_name TEXT,
                            field_name TEXT NOT NULL,
                            corrected_value TEXT NOT NULL,
                            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                            confidence REAL DEFAULT 0.95,
                            UNIQUE(document_hash, field_name)
                        )
                    """)
                    conn.commit()

                # Try again with minimal database
                from learning.database import LearningDatabase  # âœ… Correct

                self.learning_db = LearningDatabase(
                    db_path=str(learning_db_path))
                print("âœ… Minimal learning database created and connected")

            except Exception as fallback_error:
                print(f"âŒ Fallback database creation failed: {fallback_error}")
                self.learning_db = None

        # Initialize pattern learner
        try:
            from learning.pattern_learner import PatternLearner
            self.pattern_learner = PatternLearner(
                db_path=str(learning_db_path))
            print("âœ… Pattern learner initialized")
        except Exception as e:
            print(f"âŒ Pattern learner initialization failed: {e}")
            self.pattern_learner = None

    def _init_additional_components(self):
        """Initialize additional application components."""
        try:
            # Performance metrics
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {
                    'total_documents': 0,
                    'average_accuracy': 0.75,
                    'total_corrections': 0,
                    'extraction_time_ms': 0,
                    'last_updated': datetime.now().isoformat()
                }

            # Initialize other required session variables
            session_defaults = {
                'corrections_history': [],
                'extracted_tables': [],
                'original_tables': [],
                'current_document_hash': '',
                'current_pdf_name': '',
                'processing_complete': False
            }

            for key, default_value in session_defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

            print("Additional components initialized")

        except Exception as e:
            print(f"Â Error initializing additional components: {e}")

    def _verify_system_health(self):
        """Comprehensive system health verification."""
        print("\nÂ SYSTEM HEALTH CHECK:")
        print("=" * 50)

        components = {
            'Adaptive Extractor': self.adaptive_extractor,
            'Pattern Learner': self.pattern_learner,
            'Learning Database': self.learning_db
        }

        working_count = 0
        for name, component in components.items():
            if component:
                try:
                    # Test component functionality
                    if hasattr(component, 'get_learning_statistics'):
                        component.get_learning_statistics()
                    elif hasattr(component, 'get_table_correction_stats'):
                        component.get_table_correction_stats()

                    print(f" {name}: HEALTHY")
                    working_count += 1
                except Exception as e:
                    print(f"Ã°Å¸Å¸Â¡ {name}: DEGRADED ({str(e)[:30]}...)")
                    working_count += 0.5
            else:
                print(f" {name}: UNAVAILABLE")

        # Overall status
        total_components = len(components)
        if working_count >= total_components:
            status = "FULLY OPERATIONAL"
            icon = ""
        elif working_count >= total_components * 0.7:
            status = "OPERATIONAL"
            icon = "Å“â€¦"
        elif working_count >= total_components * 0.3:
            status = "LIMITED"
            icon = "Â"
        else:
            status = "CRITICAL"
            icon = "ÂÅ’"

        print(
            f"{icon} SYSTEM STATUS: {status} ({working_count}/{total_components} components)")
        print("=" * 50)

    def setup_session_state(self):
        """Enhanced session state setup with all required variables."""
        session_defaults = {
            'initialized': True,
            'extracted_fields': {},
            'confidence_scores': {},
            'extracted_tables': [],
            'processing_complete': False,
            'corrections_history': [],
            'current_pdf_name': "",
            'extracted_text': "",
            'original_tables': [],
            'current_document_hash': "",
            'current_pdf_path': "",
            'multi_doc_data': {},
            'performance_metrics': {
                'total_documents': 0,
                'average_accuracy': 0.75,
                'total_corrections': 0,
                'extraction_time_ms': 0,
                'last_updated': datetime.now().isoformat()
            }
        }

        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def generate_document_hash(file_bytes: bytes) -> str:
        """Generate consistent document hash for tracking."""
        return hashlib.sha256(file_bytes).hexdigest()

    def process_document_with_adaptive_learning(self, uploaded_file) -> bool:
        """Enhanced document processing with adaptive learning, pattern matching, and correction loading."""
        try:
            st.write("ðŸ” DEBUG: Starting document processing...")

            # Read file content
            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                st.error("âŒ Empty file uploaded")
                return False

            st.write(f"ðŸ” DEBUG: File size: {len(file_bytes)} bytes")

            # Generate document hash early for correction loading
            document_hash = self.generate_document_hash(file_bytes)
            st.session_state.current_document_hash = document_hash
            st.write(f"ðŸ” DEBUG: Document hash: {document_hash[:8]}...")

            # âœ… NEW: Load saved corrections first
            saved_corrections = {}
            try:
                if self.learning_db:
                    saved_corrections = self.learning_db.get_corrections_for_document(
                        document_hash)
                    if saved_corrections:
                        st.info(
                            f"ðŸ“‹ Found {len(saved_corrections)} saved corrections for this document")
                        for field, value in saved_corrections.items():
                            st.write(f"- {field}: {value}")
                    st.write(
                        f"ðŸ” DEBUG: Loaded {len(saved_corrections)} saved corrections")
            except Exception as e:
                st.warning(f"âš ï¸ Could not load saved corrections: {e}")
                saved_corrections = {}

            temp_suffix = f"_{int(time.time() * 1000)}.pdf"
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=temp_suffix, prefix="pdf_extract_")
            try:
                with os.fdopen(temp_fd, 'wb') as tmp_file:
                    tmp_file.write(file_bytes)
                # ... your processing code ...
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            try:
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)

                st.write("ðŸ” DEBUG: Temporary file created successfully")

                # 1. Extract tables with adaptive learning
                st.write("ðŸ” DEBUG: Starting table extraction...")
                tables = self.extract_tables_from_pdf(temp_path)
                
                # Store tables in session state immediately
                if tables:
                    st.session_state.extracted_tables = tables
                    st.write("✅ DEBUG: Tables stored in session state")
                    
                    # Debug: Show table structure
                    for i, table in enumerate(tables[:2]):  # Show first 2 tables
                        st.write(f"🔍 DEBUG: Table {i+1} structure:")
                        if isinstance(table, dict):
                            if 'headers' in table and 'rows' in table:
                                st.write(f"  - Headers: {table['headers'][:3]}...")  # First 3 headers
                                st.write(f"  - Rows: {len(table['rows'])}")
                                if table['rows']:
                                    st.write(f"  - First row: {table['rows'][0][:3]}...")  # First 3 cells
                            else:
                                st.write(f"  - Keys: {list(table.keys())}")
                        else:
                            st.write(f"  - Type: {type(table)}")
                else:
                    st.session_state.extracted_tables = []
                    st.write("⚠️ DEBUG: No tables to store")

                st.write(
                    f"ðŸ” DEBUG: Tables extracted: {len(tables) if tables else 0}")

                # 2. Extract text for field processing
                st.write("ðŸ” DEBUG: Starting text extraction...")
                try:
                    # Try to use the imported function, fallback to our function if not available
                    if 'extract_text_from_pdf' in globals():
                        text, is_scanned = extract_text_from_pdf(temp_path)
                    else:
                        text, is_scanned = extract_text_from_pdf_fallback(temp_path)
                    st.write(
                        f"ðŸ” DEBUG: Text extracted: {len(text) if text else 0} characters")

                    if text:
                        st.write(f"ðŸ” DEBUG: Text preview: {text[:200]}...")
                    else:
                        st.error("âŒ DEBUG: No text extracted!")

                except Exception as text_error:
                    st.error(f"âŒ DEBUG: Text extraction failed: {text_error}")
                    text = ""
                    is_scanned = False

                # 3. âœ… ENHANCED: Extract individual fields with learning
                st.write(
                    "ðŸ” DEBUG: Starting enhanced field extraction with learning...")
                try:
                    if text and len(text.strip()) > 0:
                        # Detect language
                        try:
                            # Try to use the imported function, fallback to our function if not available
                            if 'detect_language' in globals():
                                language = detect_language(text)
                            else:
                                language = detect_language_fallback(text)
                            st.write(f"ðŸ” DEBUG: Language detected: {language}")
                        except Exception as lang_error:
                            st.warning(
                                f"âš ï¸ DEBUG: Language detection failed: {lang_error}")
                            language = 'de'  # Default to German

                        # Extract fields using original extractor first
                        st.write("ðŸ” DEBUG: Initializing PDFFieldExtractor...")
                        base_extracted_fields = {}

                        try:
                            if FIELD_EXTRACTION_AVAILABLE:
                                extractor = PDFFieldExtractor(
                                    receiver_company="")
                                st.write(
                                    "ðŸ” DEBUG: PDFFieldExtractor created successfully")
                                all_extracted_fields = extractor.extract_fields(
                                    text, language=language)
                                st.write(
                                    f"ðŸ” DEBUG: Base extracted fields: {all_extracted_fields}")

                                base_extracted_fields = {
                                    'date': all_extracted_fields.get('date', ''),
                                    'angebot': all_extracted_fields.get('angebot', ''),
                                    'company_name': all_extracted_fields.get('company_name', ''),
                                    'sender_address': all_extracted_fields.get('sender_address', '')
                                }
                            else:
                                st.write(
                                    "ðŸ” DEBUG: Using fallback field extraction...")
                                base_extracted_fields = self._fallback_field_extraction(
                                    text)

                            st.write(
                                f"ðŸ” DEBUG: Base fields before learning: {base_extracted_fields}")

                            # âœ… NEW: Enhance with learned patterns
                            enhanced_fields = {}
                            for field_name, base_value in base_extracted_fields.items():
                                try:
                                    # Apply learned patterns to improve extraction
                                    enhanced_value = self.extract_field_with_learning(
                                        text, field_name, base_value)
                                    enhanced_fields[field_name] = enhanced_value

                                    if enhanced_value != base_value:
                                        st.write(
                                            f"ðŸ§  DEBUG: Enhanced {field_name}: '{base_value}' â†’ '{enhanced_value}'")
                                    else:
                                        st.write(
                                            f"ðŸ” DEBUG: No improvement for {field_name}: '{base_value}'")

                                except Exception as enhancement_error:
                                    st.warning(
                                        f"âš ï¸ Enhancement failed for {field_name}: {enhancement_error}")
                                    enhanced_fields[field_name] = base_value

                            extracted_fields = enhanced_fields
                            st.write(
                                f"ðŸ” DEBUG: Enhanced fields: {extracted_fields}")

                            # âœ… NEW: Apply saved corrections (highest priority)
                            for field_name, corrected_value in saved_corrections.items():
                                if corrected_value and corrected_value.strip():
                                    original_value = extracted_fields.get(
                                        field_name, '')
                                    extracted_fields[field_name] = corrected_value
                                    st.write(
                                        f"âœ… DEBUG: Applied saved correction {field_name}: '{original_value}' â†’ '{corrected_value}'")

                            # Calculate confidence scores
                            confidence_scores = {}
                            for field_name, field_value in extracted_fields.items():
                                if field_name in saved_corrections and saved_corrections[field_name]:
                                    # High confidence for saved corrections
                                    confidence_scores[field_name] = 0.95
                                elif field_value and field_value.strip():
                                    # Good confidence for extracted fields
                                    confidence_scores[field_name] = 0.8
                                else:
                                    # No confidence for empty fields
                                    confidence_scores[field_name] = 0.0

                            st.write(
                                f"ðŸ” DEBUG: Final confidence scores: {confidence_scores}")

                        except Exception as extractor_error:
                            st.error(
                                f"âŒ DEBUG: Field extraction failed: {extractor_error}")
                            st.write(
                                "ðŸ” DEBUG: Using simple fallback extraction...")
                            extracted_fields = self._simple_fallback_extraction(
                                text)

                            # Still apply saved corrections to fallback
                            for field_name, corrected_value in saved_corrections.items():
                                if corrected_value and corrected_value.strip():
                                    extracted_fields[field_name] = corrected_value

                            confidence_scores = {
                                field: 0.95 if field in saved_corrections and saved_corrections[field]
                                else (0.3 if value else 0.0)
                                for field, value in extracted_fields.items()
                            }

                    else:
                        st.error(
                            "âŒ DEBUG: No text available for field extraction!")
                        extracted_fields = {
                            'date': '',
                            'angebot': '',
                            'company_name': '',
                            'sender_address': ''
                        }

                        # Apply saved corrections even if no text was extracted
                        for field_name, corrected_value in saved_corrections.items():
                            if corrected_value and corrected_value.strip():
                                extracted_fields[field_name] = corrected_value

                        confidence_scores = {
                            field: 0.95 if field in saved_corrections and saved_corrections[
                                field] else 0.0
                            for field in extracted_fields.keys()
                        }

                except Exception as field_error:
                    st.error(
                        f"âŒ DEBUG: Field extraction completely failed: {field_error}")
                    # Fallback to empty fields but still apply saved corrections
                    extracted_fields = {
                        'date': '',
                        'angebot': '',
                        'company_name': '',
                        'sender_address': ''
                    }

                    # Apply saved corrections as last resort
                    for field_name, corrected_value in saved_corrections.items():
                        if corrected_value and corrected_value.strip():
                            extracted_fields[field_name] = corrected_value

                    confidence_scores = {
                        field: 0.95 if field in saved_corrections and saved_corrections[
                            field] else 0.0
                        for field in extracted_fields.keys()
                    }

                # Store ALL results in session state
                st.write("ðŸ” DEBUG: Storing results in session state...")
                st.session_state.extracted_tables = tables
                st.session_state.original_tables = copy.deepcopy(tables)
                st.session_state.extracted_fields = extracted_fields
                st.session_state.confidence_scores = confidence_scores
                st.session_state.extracted_text = text
                st.session_state.current_pdf_path = temp_path
                st.session_state.current_pdf_name = uploaded_file.name

                # Save document metadata
                if self.learning_db:
                    try:
                        self.learning_db.save_document_metadata(
                            document_hash, uploaded_file.name, len(file_bytes)
                        )
                    except Exception as metadata_error:
                        st.warning(
                            f"âš ï¸ Could not save document metadata: {metadata_error}")

                # Update performance metrics
                st.session_state.performance_metrics['total_documents'] += 1
                st.session_state.performance_metrics['last_updated'] = datetime.now(
                ).isoformat()

                # Show results for BOTH tables and fields
                field_count = len(
                    [v for v in extracted_fields.values() if v.strip()])
                table_count = len(tables) if tables else 0
                correction_count = len(
                    [v for v in saved_corrections.values() if v.strip()])

                st.write(
                    f"ðŸ” DEBUG: Final counts - Tables: {table_count}, Fields: {field_count}, Corrections applied: {correction_count}")

                # Enhanced success messages
                success_parts = []
                if table_count > 0:
                    success_parts.append(f"{table_count} tables")
                if field_count > 0:
                    success_parts.append(f"{field_count} fields")
                if correction_count > 0:
                    success_parts.append(
                        f"{correction_count} saved corrections applied")

                if success_parts:
                    st.success(
                        f"âœ… Successfully extracted {' and '.join(success_parts)} with adaptive learning!")
                    if correction_count > 0:
                        st.info(
                            "ðŸ§  This document had previous corrections that were automatically applied")
                else:
                    st.warning("âš ï¸ No tables or fields found in the document")
                    return False

                return True

            except Exception as e:
                st.error(f"âŒ DEBUG: Error during extraction: {e}")
                import traceback
                st.code(traceback.format_exc())
                return False

            finally:
                # Always clean up temporary file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        st.write("ðŸ” DEBUG: Temporary file cleaned up")
                    except:
                        pass  # Ignore cleanup errors

        except Exception as e:
            st.error(f"âŒ DEBUG: Error processing document: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def extract_field_with_learning(self, text, field_name, base_value):
        """Extract field with adaptive learning patterns."""
        try:
            # If we have a pattern learner, use it to enhance extraction
            if hasattr(self, 'pattern_learner') and self.pattern_learner:
                # Get learned patterns for this field type
                learned_patterns = self.pattern_learner.get_learned_patterns(field_name)
                
                if learned_patterns:
                    # Try to apply learned patterns to improve extraction
                    for pattern_info in learned_patterns:
                        pattern = pattern_info['pattern']
                        confidence = pattern_info['confidence']
                        
                        if confidence > 0.7:  # Only use high-confidence patterns
                            try:
                                import re
                                matches = re.findall(pattern, text, re.IGNORECASE)
                                if matches:
                                    # Use the first match if it's better than base value
                                    enhanced_value = matches[0] if isinstance(matches[0], str) else str(matches[0])
                                    if len(enhanced_value) > len(base_value) or (not base_value and enhanced_value):
                                        # Update pattern performance
                                        self.pattern_learner.update_pattern_performance(field_name, pattern, True)
                                        return enhanced_value
                            except Exception as e:
                                # Pattern failed, update performance
                                self.pattern_learner.update_pattern_performance(field_name, pattern, False)
                                continue
            
            # If no learned patterns or they didn't help, return base value
            return base_value
            
        except Exception as e:
            print(f"Error in extract_field_with_learning: {e}")
            return base_value

    def save_adaptive_table_corrections(self, table_index: int) -> bool:
        """Save table corrections with enhanced error handling."""
        try:
            if not hasattr(st.session_state, 'extracted_tables'):
                st.error("No tables available for correction saving")
                return False

            original_tables = st.session_state.get('original_tables', [])
            current_tables = st.session_state.get('extracted_tables', [])

            if not original_tables or not current_tables:
                st.error("Missing original or current table data")
                return False

            # Save corrections using adaptive learning
            corrections_saved = 0
            if self.adaptive_extractor:
                corrections_saved = self.adaptive_extractor.bulk_save_corrections(
                    st.session_state.get('current_pdf_path'),
                    original_tables,
                    current_tables
                )

            if corrections_saved > 0:
                st.success(
                    f"Saved {corrections_saved} corrections and learned new patterns!")
                st.balloons()

                # Update performance metrics
                st.session_state.performance_metrics['total_corrections'] += corrections_saved
                st.session_state.performance_metrics['last_updated'] = datetime.now(
                ).isoformat()

                return True
            else:
                st.info("No corrections detected to save")
                return False

        except Exception as e:
            st.error(f"ÂÅ’ Error saving corrections: {e}")
            logger.error(f"Correction saving error: {e}")
            return False

    def _render_extracted_tables(self):
        """Render extracted tables with editing capabilities."""
        tables = st.session_state.get('extracted_tables', [])

        for table_idx, table in enumerate(tables):
            confidence = table.get('overall_confidence', 0)

            with st.expander(f"â€¹ Table {table_idx + 1} - Confidence: {confidence:.1f}%"):

                # Show learning status
                if table.get('learning_applied'):
                    st.success(
                        " Learning corrections have been applied to this table!")

                # Extract and edit headers
                headers = table.get('headers', [])
                if not headers and table.get('data') and len(table['data']) > 0:
                    headers = table['data'][0]

                if headers:
                    st.write("**Edit Headers:**")
                    edited_headers = self._render_header_editor(
                        table_idx, headers)

                    # Update table with edited headers
                    if 'data' in table and len(table['data']) > 0:
                        table['data'][0] = edited_headers
                    table['headers'] = edited_headers

                    # Display table data
                    self._render_table_data(table, edited_headers)

                    # Save corrections button
                    if st.button(
                        f"Ã°Å¸â€™Â¾ Save Corrections for Table {table_idx + 1}",
                        key=f"save_{table_idx}_{st.session_state.current_pdf_name}",
                        help="Save your corrections to improve future extractions"
                    ):
                        self.save_adaptive_table_corrections(table_idx)

    def _render_header_editor(self, table_idx: int, headers: List[str]) -> List[str]:
        """Render header editing interface."""
        edited_headers = []
        # Limit UI columns for better display
        num_columns = min(len(headers), 6)

        if num_columns > 0:
            cols = st.columns(num_columns)

            for col_idx, (col, header) in enumerate(zip(cols, headers[:num_columns])):
                with col:
                    edited_header = st.text_input(
                        f"Column {col_idx + 1}",
                        value=str(header),
                        key=f"header_{table_idx}_{col_idx}_{st.session_state.current_pdf_name}"
                    )
                    edited_headers.append(edited_header)

            # Add remaining headers if any
            edited_headers.extend(headers[num_columns:])

        return edited_headers

    def _render_table_data(self, table: Dict, headers: List[str]):
        """Render table data with proper formatting."""
        if table.get('data') and len(table['data']) > 1:
            try:
                df = pd.DataFrame(table['data'][1:],
                                  columns=headers if headers else None)
                st.dataframe(df, use_container_width=True)

                # Show table statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Confidence",
                              f"{table.get('overall_confidence', 0):.1f}%")

            except Exception as e:
                st.error(f"Error displaying table: {e}")
                st.write("Raw table data:")
                st.json(table.get('data', [])[:5])  # Show first 5 rows

    def render_header(self):
        """Render enhanced application header."""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.title(" Adaptive PDF Data Extraction System")
            st.markdown(
                "**High-accuracy extraction with continuous learning**")

        with col2:
            metrics = st.session_state.performance_metrics
            st.metric("Documents Processed", metrics['total_documents'])

        with col3:
            st.metric("Average Accuracy", f"{metrics['average_accuracy']:.1%}")

        st.divider()

    def render_adaptive_table_extraction(self):
        """Enhanced table extraction UI with adaptive learning and field display - FIXED duplicate widget IDs."""
        st.header("ðŸ“Š Adaptive Table Extraction")

        uploaded_files = st.file_uploader("Upload PDFs", type=[
            "pdf"], accept_multiple_files=True, key="adaptive_table_extraction_uploader")

        if uploaded_files:
            # âœ… FIX: Generate unique session ID to prevent duplicate keys
            import time
            import uuid
            session_id = str(int(time.time() * 1000))  # Millisecond timestamp

            for file_idx, uploaded_file in enumerate(uploaded_files):
                file_content = uploaded_file.getvalue()
                document_hash = self.generate_document_hash(file_content)

                # âœ… FIXED: Create truly unique key prefix for each file
                unique_key_prefix = f"{session_id}_{file_idx}_{document_hash[:8]}"

                st.write(f"**Processing:** {uploaded_file.name}")
                
                # Save uploaded file temporarily for fallback processing
                temp_pdf_path = None
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file_content)
                    temp_pdf_path = tmp_file.name

                # Single fallback button section
                st.markdown("#### 🔄 Fallback Extraction")
                
                fallback_disabled = st.session_state.get(f'fallback_running_{document_hash}', False)
                if st.button(
                    "🚀 Run test.py Fallback (Full Extraction)", 
                    key=f"fallback_full_{unique_key_prefix}",
                    disabled=fallback_disabled,
                    use_container_width=True,
                    help="Run test.py on the entire PDF to extract both fields and tables independently"
                ):
                    st.session_state[f'fallback_running_{document_hash}'] = True
                    
                    with st.spinner("🔄 Running complete fallback extraction via test.py..."):
                        result = self.run_test_py_fallback(temp_pdf_path, 'full')
                        self.add_run_log_entry(result['command'], result['success'], result['stdout'], result['stderr'])
                        
                        if result['success'] and result['data']:
                            # Parse the complete result
                            extracted_data = result['data']
                            
                            # Extract fields and tables from the result
                            fields_data = {}
                            tables_data = []
                            
                            if isinstance(extracted_data, dict):
                                # Handle structured response with fields and tables
                                if 'fields' in extracted_data:
                                    fields_data = extracted_data['fields']
                                if 'tables' in extracted_data:
                                    tables_data = extracted_data['tables']
                                
                                # If no explicit fields/tables structure, treat the whole dict as fields
                                if not fields_data and not tables_data:
                                    fields_data = extracted_data
                            
                            # Update session state with both fields and tables
                            if fields_data:
                                st.session_state.extracted_fields = fields_data
                                st.session_state.confidence_scores = {k: 0.8 for k in fields_data.keys()}
                                
                                # Save fields to file
                                fields_saved_path = self.save_fallback_results(uploaded_file.name, fields_data, 'fields')
                                
                                # Display extracted fields immediately
                                st.subheader("📋 Fallback Extracted Fields")
                                field_df_data = []
                                for field_name, field_value in fields_data.items():
                                    field_df_data.append({
                                        "Field": field_name,
                                        "Value": str(field_value) if field_value else "(empty)",
                                        "Source": "test.py fallback"
                                    })
                                
                                if field_df_data:
                                    df = pd.DataFrame(field_df_data)
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            if tables_data:
                                st.session_state.extracted_tables = tables_data
                                
                                # Save tables to file
                                tables_saved_path = self.save_fallback_results(uploaded_file.name, tables_data, 'tables')
                                
                                # Display tables immediately
                                st.subheader("📊 Fallback Extracted Tables")
                                self.display_tables_beautifully(tables_data)
                            
                            # Show success message
                            success_parts = []
                            if fields_data:
                                success_parts.append(f"{len(fields_data)} fields")
                            if tables_data:
                                success_parts.append(f"{len(tables_data)} tables")
                            
                            if success_parts:
                                success_msg = f"✅ Extracted {' and '.join(success_parts)} from fallback"
                                st.success(success_msg)
                                st.toast("Fallback extraction completed", icon="✅")
                            else:
                                st.warning("⚠️ Fallback completed but no data was extracted")
                            
                            # Refresh the main UI
                            st.rerun()
                            
                        else:
                            error_msg = result['error'] or "Unknown error"
                            if result['stderr']:
                                error_msg += f" - {result['stderr'][:200]}"
                            st.error(f"❌ Fallback failed: {error_msg}")
                            st.toast("Fallback extraction failed", icon="❌")
                    
                    st.session_state[f'fallback_running_{document_hash}'] = False
                
                # Add comprehensive export button
                st.markdown("#### 📊 Export Options")
                if st.button(
                    "📊 Export All Data to Excel", 
                    key=f"export_comprehensive_{unique_key_prefix}",
                    use_container_width=True,
                    help="Export both extracted fields and tables to a comprehensive Excel file with multiple sheets"
                ):
                    # Get current data
                    current_fields = getattr(st.session_state, 'extracted_fields', {})
                    current_tables = getattr(st.session_state, 'extracted_tables', [])
                    
                    if current_fields or current_tables:
                        self.export_comprehensive_excel(
                            fields=current_fields,
                            tables=current_tables,
                            filename_prefix=f"comprehensive_{uploaded_file.name.split('.')[0]}"
                        )
                    else:
                        st.warning("No data available to export. Please extract fields or tables first.")

                # Process with adaptive learning
                if self.process_document_with_adaptive_learning(uploaded_file):

                    # âœ… DISPLAY EXTRACTED FIELDS FIRST
                    if hasattr(st.session_state, 'extracted_fields') and st.session_state.extracted_fields:
                        st.subheader("ðŸ“‹ Extracted Fields")

                        field_labels = {
                            "date": "ðŸ“… Date",
                            "angebot": "ðŸ“„ Angebot Number",
                            "company_name": "ðŸ¢ Company Name",
                            "sender_address": "ðŸ“ Sender Address"
                        }

                        # Create two columns for better layout
                        col1, col2 = st.columns(2)

                        # Left column: Date and Angebot
                        with col1:
                            for field_name in ["date", "angebot"]:
                                value = st.session_state.extracted_fields.get(
                                    field_name, "")
                                confidence = st.session_state.confidence_scores.get(
                                    field_name, 0)

                                edited_value = st.text_input(
                                    field_labels[field_name],
                                    value=value,
                                    # âœ… FIXED
                                    key=f"adaptive_field_{field_name}_{unique_key_prefix}",
                                    help=f"Confidence: {confidence:.1%}"
                                )

                                # Update session state with edited values
                                st.session_state.extracted_fields[field_name] = edited_value

                        # Right column: Company Name and Address
                        with col2:
                            for field_name in ["company_name", "sender_address"]:
                                value = st.session_state.extracted_fields.get(
                                    field_name, "")
                                confidence = st.session_state.confidence_scores.get(
                                    field_name, 0)

                                edited_value = st.text_input(
                                    field_labels[field_name],
                                    value=value,
                                    # âœ… FIXED
                                    key=f"adaptive_field_{field_name}_{unique_key_prefix}",
                                    help=f"Confidence: {confidence:.1%}"
                                )

                                # Update session state with edited values
                                st.session_state.extracted_fields[field_name] = edited_value

                        # Action buttons for fields
                        col_save, col_export = st.columns(2)

                        with col_save:
                            if st.button(f"ðŸ’¾ Save Field Corrections",
                                         # âœ… FIXED
                                         key=f"save_fields_{unique_key_prefix}",
                                         use_container_width=True):
                                # Save corrections to database
                                if self.save_corrections_to_database():
                                    st.success(
                                        "âœ… Field corrections saved successfully!")

                                    # Learn from corrections
                                    if hasattr(self, 'pattern_learner') and self.pattern_learner:
                                        for field_name, corrected_value in st.session_state.extracted_fields.items():
                                            if corrected_value:
                                                context = st.session_state.get(
                                                    'extracted_text', '')[:1000]
                                                self.pattern_learner.learn_from_feedback(
                                                    field_name, corrected_value, context)

                        with col_export:
                            if st.button(f"ðŸ“Š Export Fields to Excel",
                                         # âœ… FIXED
                                         key=f"export_adaptive_fields_{unique_key_prefix}",
                                         use_container_width=True):
                                # Use comprehensive export with both fields and tables
                                self.export_comprehensive_excel(
                                    fields=st.session_state.extracted_fields,
                                    tables=getattr(st.session_state, 'extracted_tables', []),
                                    filename_prefix=f"pdf_extraction_{uploaded_file.name.split('.')[0]}"
                                )

                        # Add parameter controls
                        st.markdown("---")
                        self.render_parameter_controls(st.session_state.extracted_fields, document_hash)

                        # Separator between fields and tables
                        st.markdown("---")

                    # âœ… DISPLAY EXTRACTED TABLES SECOND
                    if hasattr(st.session_state, 'extracted_tables') and st.session_state.extracted_tables:
                        st.subheader("ðŸ“Š Extracted Tables")

                        for table_idx, table in enumerate(st.session_state.extracted_tables):
                            with st.expander(f"ðŸ“‹ Table {table_idx + 1} - Confidence: {table.get('overall_confidence', 0):.1f}%"):

                                # Show learning metadata
                                if table.get('learning_applied'):
                                    st.success(
                                        "ðŸ§  Learning corrections have been applied to this table!")

                                # Edit table headers
                                st.write("**Edit Headers:**")
                                headers = table.get('headers', [])
                                if not headers and table.get('data'):
                                    headers = table['data'][0]

                                edited_headers = []
                                if headers:
                                    # Limit to 6 columns for UI display
                                    cols = st.columns(min(len(headers), 6))
                                    for col_idx, (col, header) in enumerate(zip(cols, headers)):
                                        with col:
                                            edited_header = st.text_input(
                                                f"Column {col_idx + 1}",
                                                value=str(header),
                                                # âœ… FIXED
                                                key=f"adaptive_header_{table_idx}_{col_idx}_{unique_key_prefix}"
                                            )
                                            edited_headers.append(
                                                edited_header)

                                    # Update headers in session state
                                    if 'data' in table and len(table['data']) > 0:
                                        table['data'][0] = edited_headers
                                    table['headers'] = edited_headers

                                # Display table data
                                if table.get('data') and len(table['data']) > 1:
                                    try:
                                        df = pd.DataFrame(
                                            table['data'][1:],
                                            columns=edited_headers if edited_headers else None
                                        )
                                        st.dataframe(
                                            df, use_container_width=True)

                                        # Show table statistics
                                        table_col1, table_col2, table_col3 = st.columns(
                                            3)
                                        with table_col1:
                                            st.metric("Rows", len(df))
                                        with table_col2:
                                            st.metric(
                                                "Columns", len(df.columns))
                                        with table_col3:
                                            st.metric(
                                                "Confidence", f"{table.get('overall_confidence', 0):.1f}%")

                                    except Exception as e:
                                        st.error(
                                            f"Error displaying table: {e}")
                                        st.write("Raw table data:")
                                        # Show first 5 rows
                                        st.json(table.get('data', [])[:5])

                                # Save table corrections button
                                if st.button(f"ðŸ’¾ Save Table {table_idx + 1} Corrections",
                                             # âœ… FIXED
                                             key=f"save_adaptive_table_{table_idx}_{unique_key_prefix}",
                                             use_container_width=True):
                                    if self.save_adaptive_table_corrections(table_idx):
                                        st.success(
                                            f"âœ… Table {table_idx + 1} corrections saved!")

                    # Show message if no data extracted
                    if (not hasattr(st.session_state, 'extracted_fields') or
                        not any(st.session_state.extracted_fields.values())) and \
                        (not hasattr(st.session_state, 'extracted_tables') or
                            not st.session_state.extracted_tables):
                        st.info(
                            "â„¹ï¸ No fields or tables were extracted from this document. Try a different PDF or check the document quality.")

                    st.markdown("---")  # Separator between documents
                    
                    # Clean up temporary file
                    if temp_pdf_path and os.path.exists(temp_pdf_path):
                        try:
                            os.unlink(temp_pdf_path)
                        except Exception:
                            pass  # Ignore cleanup errors

            # Add run log panel after all files are processed
            self.render_run_log_panel()

    def render_sidebar(self):
        """Render enhanced sidebar with system status."""
        with st.sidebar:
            st.header("ðŸ”§ System Control")

            # System status indicators
            st.subheader("System Status")

            if self.adaptive_extractor:
                st.success("âœ… Adaptive Learning: Active")
            else:
                st.error("âŒ Adaptive Learning: Inactive")

            if self.learning_db:
                st.success("âœ… Database: Connected")
            else:
                st.error("âŒ Database: Disconnected")

            # Quick stats
            st.subheader("ðŸ“Š Quick Stats")
            metrics = st.session_state.performance_metrics
            st.metric("Documents", metrics['total_documents'])
            st.metric("Corrections", metrics['total_corrections'])

            # Quick actions
            st.subheader("âš¡ Quick Actions")
            import uuid
            if st.button("ðŸ”„ Refresh System", key=f"refresh_system_{uuid.uuid4().hex[:12]}", use_container_width=True):
                st.rerun()

    def safe_method_call(self, method, *args, fallback_result=None, **kwargs):
        """Safely call methods with fallback on failure."""
        try:
            return method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Method {method.__name__} failed: {e}")
            return fallback_result

    def diagnose_database_connection(self):
        """Complete database connection diagnosis."""
        st.header("ðŸ” Database Connection Diagnosis")

        # Check 1: Database instance
        st.subheader("1. Database Instance Check")
        if hasattr(self, 'learning_db') and self.learning_db:
            st.success("âœ… learning_db instance exists")
            db_path = self.learning_db.db_path
            st.write(f"**Database path:** `{db_path}`")
        else:
            st.error("âŒ learning_db instance missing")
            return

        # Check 2: File system
        st.subheader("2. File System Check")
        import os
        from pathlib import Path

        abs_path = os.path.abspath(db_path)
        parent_dir = Path(db_path).parent

        st.write(f"**Absolute path:** `{abs_path}`")
        st.write(f"**Parent directory:** `{parent_dir}`")

        if parent_dir.exists():
            st.success(f"âœ… Directory exists: {parent_dir}")
        else:
            st.error(f"âŒ Directory missing: {parent_dir}")
            if st.button("ðŸ“ Create Directory"):
                parent_dir.mkdir(parents=True, exist_ok=True)
                st.success("âœ… Directory created!")
                st.experimental_rerun()

        if os.path.exists(db_path):
            st.success(f"âœ… Database file exists")
            file_size = os.path.getsize(db_path)
            st.write(f"**File size:** {file_size} bytes")
        else:
            st.error(f"âŒ Database file missing: {db_path}")
            if st.button("ðŸ—ƒï¸ Create Database"):
                try:
                    self.learning_db._init_database()
                    st.success("âœ… Database created!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"âŒ Database creation failed: {e}")

        # Check 3: Database connectivity
        st.subheader("3. Database Connectivity Test")
        try:
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # List tables
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                if tables:
                    st.success(
                        f"âœ… Connected! Found tables: {', '.join(tables)}")

                    # Check corrections table
                    if 'corrections' in tables:
                        cursor.execute("SELECT COUNT(*) FROM corrections")
                        correction_count = cursor.fetchone()[0]
                        st.metric("Stored Corrections", correction_count)

                        if correction_count > 0:
                            st.write("**Recent corrections:**")
                            cursor.execute("""
                                SELECT field_name, corrected_value, timestamp 
                                FROM corrections 
                                ORDER BY timestamp DESC 
                                LIMIT 5
                            """)
                            for field, value, timestamp in cursor.fetchall():
                                st.write(
                                    f"- **{field}:** {value} ({timestamp})")
                    else:
                        st.warning("âš ï¸ 'corrections' table missing")
                else:
                    st.warning("âš ï¸ No tables found - database may be empty")

        except Exception as e:
            st.error(f"âŒ Connection failed: {e}")
            st.code(str(e))

        # Check 4: Session state
        st.subheader("4. Session State Check")
        if hasattr(st.session_state, 'current_document_hash'):
            st.write(
                f"**Current document hash:** {st.session_state.current_document_hash[:8]}...")
        else:
            st.warning("âš ï¸ No current document hash in session")

        if hasattr(st.session_state, 'extracted_fields'):
            field_count = len(
                [v for v in st.session_state.extracted_fields.values() if v])
            st.write(f"**Extracted fields:** {field_count}")
        else:
            st.warning("âš ï¸ No extracted fields in session")

    def test_adaptive_system(self):
        """Test the adaptive learning system."""
        st.subheader("Ã°Å¸Â§Âª System Test")

        if st.button("Test Adaptive System", key="test_adaptive_system"):
            results = {}

            # Test adaptive extractor
            if self.adaptive_extractor:
                try:
                    stats = self.adaptive_extractor.get_learning_statistics()
                    results['adaptive_extractor'] = f"Working - {stats.get('patterns_learned', 0)} patterns learned"
                except Exception as e:
                    results['adaptive_extractor'] = f"ÂÅ’ Error: {e}"
            else:
                results['adaptive_extractor'] = "ÂÅ’ Not initialized"

            # Test learning database
            if self.learning_db:
                try:
                    test_stats = self.learning_db.get_table_correction_stats()
                    results['learning_db'] = f"Working - {test_stats.get('total_table_corrections', 0)} corrections"
                except Exception as e:
                    results['learning_db'] = f"ÂÅ’ Error: {e}"
            else:
                results['learning_db'] = "ÂÅ’ Not initialized"

            # Test pattern learner
            if self.pattern_learner:
                results['pattern_learner'] = "Working"
            else:
                results['pattern_learner'] = " Not initialized"

            # Display results
            for component, status in results.items():
                st.write(f"**{component}:** {status}")

    def load_saved_documents_from_database(self):
        """Load all saved documents from database into session state for UI display."""
        if 'multi_doc_data' not in st.session_state:
            st.session_state['multi_doc_data'] = {}

        if self.learning_db:
            try:
                # Use consistent path - THIS IS THE KEY FIX
                from pathlib import Path
                import sqlite3
                db_path = Path(__file__).parent.parent / \
                    "learning" / "learning.db"
                # Changed from hardcoded path
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute("""
                        SELECT DISTINCT document_hash, pdf_name 
                        FROM document_metadata 
                        ORDER BY last_processed DESC
                    """)

                    for doc_hash, pdf_name in cursor.fetchall():
                        corrections = self.learning_db.get_corrections_for_document(
                            doc_hash)

                        st.session_state['multi_doc_data'][doc_hash] = {
                            'filename': pdf_name,
                            'fields': corrections,
                            'confidence': {},
                            'tables': [],
                            'text': '',
                        }

                print(
                    f"Loaded {len(st.session_state['multi_doc_data'])} documents from database")

            except Exception as e:
                print(f"ÂÅ’ Error loading documents from database: {e}")

    def render_document_processing(self):
        """Enhanced document processing UI with persistent corrections."""
        st.header("â€ž Upload and Process PDF Documents")

        # Load saved documents from database into UI
        self.load_saved_documents_from_database()

        # DEBUG: Check if data was loaded
        st.write("**DEBUG INFO:**")
        st.write(
            f"multi_doc_data keys: {list(st.session_state.get('multi_doc_data', {}).keys())}")
        st.write(
            f"Total documents loaded: {len(st.session_state.get('multi_doc_data', {}))}")
        if st.session_state.get('multi_doc_data'):
            for hash_key, data in st.session_state['multi_doc_data'].items():
                st.write(
                    f"- {data['filename']}: {len(data.get('fields', {}))} fields")

        # Add Learning Debug Section
        st.markdown("---")
        st.subheader("Â Learning Debug Info")

        if st.button("Â Check Learning Status", key="check_learning_status"):
            import sqlite3
            from pathlib import Path
            try:
                # Use consistent path
                db_path = Path(__file__).parent.parent / \
                    "learning" / "learning.db"
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM learned_patterns")
                    count = cursor.fetchone()[0]
                    st.write(f"**Total learned patterns: {count}**")

                    if count > 0:
                        cursor = conn.execute(
                            "SELECT field_type, pattern, confidence FROM learned_patterns ORDER BY confidence DESC LIMIT 5")
                        patterns = cursor.fetchall()
                        st.write("**Top 5 learned patterns:**")
                        for field, pattern, conf in patterns:
                            st.write(
                                f"- **{field}:** `{pattern}` (confidence: {conf:.2f})")
                    else:
                        st.info(
                            "No patterns learned yet. Make some corrections to start learning!")
            except Exception as e:
                st.error(f"Error checking database: {e}")

        # Database Connection Debug
        if st.button("Â Debug Database Connection", key="debug_database_connection"):
            from pathlib import Path
            import sqlite3
            db_path = Path(__file__).parent.parent / "learning" / "learning.db"
            st.write(f"Database path: {db_path}")
            st.write(f"Database exists: {db_path.exists()}")

            if db_path.exists():
                try:
                    with sqlite3.connect(str(db_path)) as conn:
                        cursor = conn.execute(
                            "SELECT COUNT(*) FROM document_metadata")
                        doc_count = cursor.fetchone()[0]
                        st.write(f"Documents in database: {doc_count}")

                        cursor = conn.execute(
                            "SELECT COUNT(*) FROM corrections")
                        correction_count = cursor.fetchone()[0]
                        st.write(
                            f"Corrections in database: {correction_count}")

                        if doc_count > 0:
                            cursor = conn.execute(
                                "SELECT document_hash, pdf_name FROM document_metadata ORDER BY last_processed DESC LIMIT 5")
                            st.write("**Recent documents in database:**")
                            for hash_val, name in cursor.fetchall():
                                st.write(f"- {name} ({hash_val[:8]}...)")
                except Exception as e:
                    st.error(f"Database error: {e}")
            else:
                st.error("Database file does not exist!")

        # Force Load Test Data (for debugging)
        if st.button("Ã°Å¸Â§Âª Force Load Test Data", key="force_load_test_data"):
            st.session_state['multi_doc_data'] = {
                'test123': {
                    'filename': 'test_document.pdf',
                    'fields': {
                        'angebot': 'TEST-123',
                        'date': '01.01.2024',
                        'company_name': 'Test Company',
                        'sender_address': 'Test Address'
                    },
                    'confidence': {},
                    'tables': [],
                    'text': ''
                }
            }
            st.success("Test data force-loaded!")
            st.experimental_rerun()

        st.markdown("---")

        # Show Previously Processed Documents
        if st.session_state.get('multi_doc_data'):
            st.subheader("â€¹ Previously Processed Documents")

            # Document selector for saved documents
            doc_keys = list(st.session_state['multi_doc_data'].keys())
            selected_hash = st.selectbox(
                "Select a document to review/edit",
                options=doc_keys,
                format_func=lambda h: st.session_state['multi_doc_data'][h]['filename'],
                key="saved_doc_selector"
            )

            if selected_hash:
                doc_data = st.session_state['multi_doc_data'][selected_hash]

                # Display and edit fields for selected saved document
                st.subheader(
                    f"Â Saved Fields for {doc_data['filename']} (Editable)")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Field Values:**")
                    updated_fields = {}
                    field_labels = {
                        "date": "â€¦ Date",
                        "angebot": "â€ž Angebot Number",
                        "company_name": "Ã°Å¸ÂÂ¢ Company Name",
                        "sender_address": "Â Sender Address"
                    }

                    for field_name in ["date", "angebot", "company_name", "sender_address"]:
                        current_value = doc_data['fields'].get(field_name, "")

                        if isinstance(current_value, dict):
                            display_value = current_value.get('value', '')
                        else:
                            display_value = str(current_value)

                        updated_value = st.text_input(
                            field_labels.get(
                                field_name, field_name.replace('_', ' ').title()),
                            value=display_value,
                            key=f"saved_field_{field_name}_{selected_hash}",
                            help=f"Confidence: {doc_data.get('confidence', {}).get(field_name, 0):.1%}"
                        )
                        updated_fields[field_name] = updated_value

                    # Update the stored data
                    st.session_state['multi_doc_data'][selected_hash]['fields'] = updated_fields

                with col2:
                    st.write("**Actions:**")

                    # Save corrections button
                    if st.button("Ã°Å¸â€™Â¾ Save Corrections", key=f"save_saved_{selected_hash}", use_container_width=True):
                        # Set current document context for saving
                        st.session_state.extracted_fields = updated_fields
                        st.session_state.current_document_hash = selected_hash
                        st.session_state.current_pdf_name = doc_data['filename']

                        if self.save_corrections_to_database():
                            st.success("Corrections saved successfully!")

                            # Learn from corrections
                            for field_name, corrected_value in updated_fields.items():
                                if corrected_value:
                                    context = st.session_state.get(
                                        'extracted_text', '')[:1000]
                                    self.pattern_learner.learn_from_feedback(
                                        field_name, corrected_value, context)

                    # Export buttons
                    if st.button("Å  Export to Excel", key=f"export_saved_{selected_hash}", use_container_width=True):
                        self.export_to_excel(updated_fields)

                    if st.button("â€¹ Copy JSON", key=f"copy_saved_{selected_hash}", use_container_width=True):
                        self.display_json_data(updated_fields)

            st.markdown("---")

        # File Upload Section for New Documents
        uploaded_files = st.file_uploader("Choose PDF files", type=[
                                          "pdf"], accept_multiple_files=True)

        # Process newly uploaded files
        if uploaded_files:
            st.subheader("Processing New Documents")

            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                document_hash = self.generate_document_hash(file_content)
                # --- LOAD FIELD CORRECTIONS ---
                saved_fields = self.learning_db.get_corrections_for_document(
                    document_hash)  # Load user corrections from DB

                # Run your normal field extraction, then overlay corrections
                extracted_fields = self.extract_fields_from_pdf(
                    uploaded_file)  # Or however you call your extraction
                # Corrected fields take priority
                extracted_fields.update(saved_fields)

                st.session_state["extracted_fields"] = extracted_fields

                # --- LOAD TABLE CORRECTIONS ---
                # Implement this DB method if not present
                if hasattr(self.learning_db, "get_tables_for_document"):
                    saved_tables = self.learning_db.get_tables_for_document(
                        document_hash)
                    st.session_state["extracted_tables"] = saved_tables
                else:
                    # Fallback: always extract fresh
                    st.session_state["extracted_tables"] = []

                # Skip if already processed
                if document_hash in st.session_state.get('multi_doc_data', {}):
                    st.info(
                        f"{uploaded_file.name} already processed. Select from 'Previously Processed Documents' above to edit.")
                    continue

                st.session_state['current_document_hash'] = document_hash
                st.session_state['current_pdf_name'] = uploaded_file.name

                # Process document with correction persistence
                if self.process_document_with_corrections(uploaded_file):
                    # Add to multi_doc_data after processing
                    st.session_state['multi_doc_data'][document_hash] = {
                        'filename': uploaded_file.name,
                        'fields': st.session_state.get('extracted_fields', {}),
                        'confidence': st.session_state.get('confidence_scores', {}),
                        'tables': st.session_state.get('extracted_tables', []),
                        'text': st.session_state.get('extracted_text', ''),
                    }

                    # Display extracted fields with edit capability
                    st.subheader(
                        f"Â Extracted Fields for {uploaded_file.name} (Editable)")
                    st.write(
                        "Review and correct the extracted information below:")

                    # Create columns for better layout
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Field Values:**")

                        # Create editable input fields
                        updated_fields = {}
                        field_labels = {
                            "date": "â€¦ Date",
                            "angebot": "â€ž Angebot Number",
                            "company_name": "Ã°Å¸ÂÂ¢ Company Name",
                            "sender_address": "Â Sender Address"
                        }

                        for field_name in ["date", "angebot", "company_name", "sender_address"]:
                            current_value = st.session_state.extracted_fields.get(
                                field_name, "")

                            # Handle different field value formats
                            if isinstance(current_value, dict):
                                display_value = current_value.get('value', '')
                            else:
                                display_value = str(current_value)

                            # Create input field
                            updated_value = st.text_input(
                                field_labels.get(
                                    field_name, field_name.replace('_', ' ').title()),
                                value=display_value,
                                key=f"new_field_{field_name}_{document_hash}",
                                help=f"Confidence: {st.session_state.confidence_scores.get(field_name, 0):.1%}"
                            )

                            updated_fields[field_name] = updated_value

                        # Update session state with edited values
                        st.session_state.extracted_fields = updated_fields
                        st.session_state['multi_doc_data'][document_hash]['fields'] = updated_fields

                    with col2:
                        st.write("**Actions:**")

                        # Save corrections button
                        if st.button("Ã°Å¸â€™Â¾ Save Corrections", key=f"save_new_{document_hash}", use_container_width=True):
                            if self.save_corrections_to_database():
                                st.success("Corrections saved successfully!")

                                # Learn from user corrections for future accuracy improvement
                                for field_name, corrected_value in updated_fields.items():
                                    if corrected_value:
                                        context = st.session_state.get(
                                            'extracted_text', '')[:1000]
                                        self.pattern_learner.learn_from_feedback(
                                            field_name, corrected_value, context)

                        # Export buttons
                        if st.button("Å  Export to Excel", key=f"export_new_{document_hash}", use_container_width=True):
                            self.export_to_excel(updated_fields)

                        if st.button("â€¹ Copy JSON", key=f"copy_new_{document_hash}", use_container_width=True):
                            self.display_json_data(updated_fields)

                    # Display document information if available
                    if self.learning_db and hasattr(st.session_state, 'current_document_hash'):
                        doc_info = self.learning_db.get_document_info(
                            st.session_state.current_document_hash)
                        if doc_info:
                            st.subheader("Å  Document Information")
                            info_col1, info_col2, info_col3, info_col4 = st.columns(
                                4)

                            info_col1.metric(
                                "Times Processed", doc_info['processing_count'])
                            info_col2.metric(
                                "Corrections Saved", doc_info['correction_count'])
                            info_col3.metric(
                                "First Processed", doc_info['first_processed'][:10])
                            info_col4.metric("Last Processed",
                                             doc_info['last_processed'][:10])

                    st.markdown("---")

        # Show message if no documents exist
        if not st.session_state.get('multi_doc_data') and not uploaded_files:
            st.info(
                "â€ž No documents processed yet. Upload a PDF file above to get started!")

    def process_document(self, uploaded_file):
        """Process uploaded document file (any format)."""
        try:
            # Import multi-format extractor
            from secure_pdf_extractor.extraction.multi_extraction import MultiFormatExtractor, detect_file_type

            file_type = detect_file_type(uploaded_file.name)

            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**File:** {uploaded_file.name}")
            with col2:
                st.info(f"**Type:** {file_type.upper()}")
            with col3:
                file_size = len(uploaded_file.getvalue())
                size_mb = file_size / (1024 * 1024)
                st.info(f"**Size:** {size_mb:.1f} MB")

            with st.spinner(f"Processing {file_type.upper()} document..."):
                # Use multi-format extractor
                extractor = MultiFormatExtractor()
                text, metadata = extractor.extract_from_file(
                    uploaded_file.getvalue(), uploaded_file.name)

                # Store metadata
                st.session_state.file_metadata = metadata

                # Try to detect language if text was extracted successfully
                if not metadata.get('error') and text.strip():
                    try:
                        # Try to use the imported function, fallback to our function if not available
                        if 'detect_language' in globals():
                            detected_language = detect_language(text)
                        else:
                            detected_language = detect_language_fallback(text)
                        language = 'de' if detected_language == 'German' else 'en'
                    except:
                        language = 'en'  # Default fallback

                    # Show success message
                    st.success(
                        f"â‚¬Â¦ Document processed successfully! Detected language: {detected_language}")

                    # Show processing details
                    with st.expander("Processing Details"):
                        for key, value in metadata.items():
                            if key not in ['error']:
                                st.write(
                                    f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    language = 'en'  # Default fallback
                    if metadata.get('error'):
                        st.warning(
                            f"ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Â¸Â Processing completed with warnings: {metadata.get('error')}")
                    else:
                        st.success("â‚¬Â¦ Document processed successfully!")

                is_scanned = metadata.get('method') == 'tesseract_ocr'

                # Extract fields using basic extraction
                extractor = PDFFieldExtractor(
                    receiver_company="", pattern_learner=pattern_learner)
                all_extracted_fields = extractor.extract_fields(
                    text, language=language)
                extracted_fields = {
                    'date': all_extracted_fields.get('date', ''),
                    'angebot': all_extracted_fields.get('angebot', ''),
                    'company_name': all_extracted_fields.get('company_name', ''), 'sender_address': all_extracted_fields.get('sender_address', '')

                }
                confidence_scores = {}
                for field_name, field_value in extracted_fields.items():
                    if field_value and field_value.strip():
                        # Default confidence
                        confidence_scores[field_name] = 0.8
                    else:
                        confidence_scores[field_name] = 0.0

                # Extract tables (only for PDF files)
                tables = []
                if file_type.lower() == 'pdf':
                    try:
                        # Save temporarily for table extraction (PDF only)
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name

                        tables = extract_tables(temp_path)

                        # Clean up temp file
                        os.unlink(temp_path)
                    except Exception as e:
                        st.warning(
                            f"ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Â¸Â Table extraction failed: {str(e)}")
                        tables = []
                else:
                    # For non-PDF files, no table extraction
                    tables = []

                # Store results
                st.session_state.extracted_fields = extracted_fields
                st.session_state.confidence_scores = confidence_scores
                st.session_state.extracted_tables = tables
                st.session_state.current_pdf_name = uploaded_file.name
                st.session_state.extracted_text = text
                st.session_state.processing_complete = True

                # Update performance metrics
                st.session_state.performance_metrics['total_documents'] += 1

                st.success(
                    f"â‚¬Â¦ {file_type.upper()} document processed successfully!")

        except Exception as e:
            st.error(f" Error processing PDF: {e}")

    def render_document_processing(self):
        """Enhanced document processing UI with persistent corrections and multi-file upload."""
        st.header("â€ž Upload and Process PDF Documents")

        uploaded_files = st.file_uploader(
            "Choose PDF files", type=["pdf"], accept_multiple_files=True, key="document_processing_uploader"
        )

        if uploaded_files:
            # Initialize storage in session state if not exists
            if 'multi_doc_data' not in st.session_state:
                st.session_state['multi_doc_data'] = {}

            # Process each uploaded file
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                document_hash = self.generate_document_hash(file_content)

                # Avoid re-processing same file multiple times in one session
                if document_hash in st.session_state['multi_doc_data']:
                    continue  # Already processed in this session

                # Save current document info in session state (per file keyed by hash)
                st.session_state['current_document_hash'] = document_hash
                st.session_state['current_pdf_name'] = uploaded_file.name

                # Process document and extract data
                success = self.process_document_with_corrections(uploaded_file)

                if success:
                    # Save extracted fields per document hash
                    st.session_state['multi_doc_data'][document_hash] = {
                        'filename': uploaded_file.name,
                        'fields': st.session_state.get('extracted_fields', {}),
                        'confidence': st.session_state.get('confidence_scores', {}),
                        'tables': st.session_state.get('extracted_tables', []),
                        'text': st.session_state.get('extracted_text', ''),
                    }
                    st.success(f"Processed {uploaded_file.name} successfully!")
                else:
                    st.error(f"Failed to process {uploaded_file.name}")

            st.markdown("---")
            st.subheader("Processed Documents")

            # UI to select which document's extracted data to view/edit
            doc_keys = list(st.session_state['multi_doc_data'].keys())
            selected_hash = st.selectbox(
                "Select a document to review/edit",
                options=doc_keys,
                format_func=lambda h: st.session_state['multi_doc_data'][h]['filename']
            )

            if selected_hash:
                doc_data = st.session_state['multi_doc_data'][selected_hash]
                extracted_fields = doc_data['fields']
                confidence_scores = doc_data['confidence']

                # Editable fields UI
                st.subheader(
                    f" Extracted Fields for {doc_data['filename']} (Editable)")
                field_labels = {
                    "date": "Date",
                    "angebot": "Angebot Number",
                    "company_name": "Company Name",
                    "sender_address": "Sender Address"
                }

                updated_fields = {}
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Field Values:**")
                    for field_name in ["date", "angebot", "company_name", "sender_address"]:
                        current_value = extracted_fields.get(field_name, "")
                        if isinstance(current_value, dict):
                            display_value = current_value.get('value', '')
                        else:
                            display_value = str(current_value)

                        updated_value = st.text_input(
                            label=field_labels.get(
                                field_name, field_name.replace('_', ' ').title()),
                            value=display_value,
                            key=f"{selected_hash}_{field_name}",
                            help=f"Confidence: {confidence_scores.get(field_name, 0):.1%}"
                        )
                        updated_fields[field_name] = updated_value

                    # Save updated back to session state
                    st.session_state['multi_doc_data'][selected_hash]['fields'] = updated_fields

                with col2:
                    st.write("**Actions:**")

                    # Save corrections button-----------------------------------------------------------------------------------------------------------------------------------

                    if st.button("Ã°Å¸â€™Â¾ Save Corrections", key=f"save_{selected_hash}", use_container_width=True):
                        # Temporarily overwrite extracted_fields to save corrections
                        st.session_state.extracted_fields = st.session_state[
                            'multi_doc_data'][selected_hash]['fields']
                        st.session_state.current_document_hash = selected_hash
                        st.session_state.current_pdf_name = doc_data['filename']

                        if self.save_corrections_to_database():
                            if 'corrections_history' not in st.session_state:
                                st.session_state.corrections_history = []

                            st.session_state.corrections_history.append({
                                'document': doc_data['filename'],
                                'hash': selected_hash[:8],
                                'timestamp': datetime.now().isoformat(),
                                'fields_corrected': len([v for v in updated_fields.values() if v])
                            })
                            st.success("Corrections saved successfully!")

                            # Learn from user corrections for future accuracy improvement
                            for field_name, corrected_value in updated_fields.items():
                                if corrected_value:
                                    context = st.session_state.get(
                                        'extracted_text', '')[:1000]
                                    self.pattern_learner.learn_from_feedback(
                                        field_name, corrected_value, context)

                    # Export to Excel button----------------------------------------------------------------------------------------------------------------------------------------
                    if st.button("Å  Export to Excel", key=f"export_excel_{selected_hash}", use_container_width=True):
                        self.export_to_excel(
                            st.session_state['multi_doc_data'][selected_hash]['fields'])

                    # Copy JSON button
                    if st.button("â€¹ Copy JSON", key=f"copy_json_{selected_hash}", use_container_width=True):
                        self.display_json_data(
                            st.session_state['multi_doc_data'][selected_hash]['fields'])

                # Document metadata display
                if self.learning_db:
                    doc_info = self.learning_db.get_document_info(
                        selected_hash)
                    if doc_info:
                        st.subheader("Å  Document Information")
                        info_col1, info_col2, info_col3, info_col4 = st.columns(
                            4)
                        info_col1.metric("Times Processed", doc_info.get(
                            'processing_count', 'N/A'))
                        info_col2.metric("Corrections Saved", doc_info.get(
                            'correction_count', 'N/A'))
                        info_col3.metric("First Processed", doc_info.get(
                            'first_processed', '')[:10])
                        info_col4.metric("Last Processed", doc_info.get(
                            'last_processed', '')[:10])

    def render_extracted_fields(self):
        """Render extracted fields with correction capabilities."""
        fields = st.session_state.extracted_fields
        confidence_scores = st.session_state.confidence_scores

        # Create form for field editing
        with st.form("field_corrections"):
            corrected_fields = {}

            required_fields = ['date', 'angebot',
                               'company_name', 'sender_address']
            filtered_fields = {k: v for k,
                               v in fields.items() if k in required_fields}

            for field_name, value in filtered_fields.items():

                col1, col2 = st.columns([3, 1])

                with col1:
                    corrected_value = st.text_input(
                        f"{field_name.replace('_', ' ').title()}",
                        value=value,
                        key=f"field_{field_name}"
                    )
                    corrected_fields[field_name] = corrected_value

                with col2:
                    confidence = confidence_scores.get(field_name, 0.0)
                    color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                    st.markdown(f"<span style='color: {color}; font-weight: bold;'>{confidence:.1%}</span>",
                                unsafe_allow_html=True)

            # Submit corrections and training options
            col1, col2 = st.columns(2)

            with col1:
                save_corrections = st.form_submit_button(
                    " Save Corrections")

            with col2:
                train_document = st.form_submit_button(
                    "  Train from Document", help="Train the model using this document's data")

            if save_corrections:
                self.save_corrections(fields, corrected_fields)

            if train_document:
                self.train_from_current_document(fields, corrected_fields)

        # Add train button outside form for when no corrections are made
        if not st.session_state.get('corrections_made', False):
            st.markdown("---")
            if st.button("  Train with Current Data", help="Train the model using the extracted data as-is"):
                self.train_from_current_document(
                    fields, corrected_fields, auto_mode=True)

    def export_to_excel(self, fields: Dict[str, str]):
        """Export extracted fields to Excel."""
        try:
            # Create export data
            data = {
                'Field': ['Date', 'Angebot Number', 'Company Name', 'Company Address'],
                'Value': [
                    fields.get('date', ''),
                    fields.get('angebot', ''),
                    fields.get('company_name', ''),
                    fields.get('sender_address', '')
                ],
                'Confidence': [
                    f"{st.session_state.confidence_scores.get('date', 0):.1%}",
                    f"{st.session_state.confidence_scores.get('angebot', 0):.1%}",
                    f"{st.session_state.confidence_scores.get('company_name', 0):.1%}",
                    f"{st.session_state.confidence_scores.get('sender_address', 0):.1%}"
                ]
            }

            # Create DataFrame and export
            df = pd.DataFrame(data)
            filename = f"extracted_4_fields_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            # Create Excel file in memory
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Extracted_Fields')
            output.seek(0)

            # Provide download
            st.download_button(
                label="Â¬â€¡Ã¯Â¸Â Download Excel File",
                data=output.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("Excel file ready for download!")

        except Exception as e:
            st.error(f"ÂÅ’ Failed to export Excel file: {e}")

    def display_json_data(self, fields: Dict[str, str]):
        """Display extracted data as JSON."""
        try:
            json_data = {
                'fields': fields,
                'tables': st.session_state.extracted_tables,
                'metadata': {
                    'filename': st.session_state.current_pdf_name,
                    'document_hash': st.session_state.current_document_hash,
                    'timestamp': datetime.now().isoformat()
                }
            }
            st.code(json.dumps(json_data, indent=2))
        except Exception as e:
            st.error(f"ÂÅ’ Error displaying JSON: {e}")

    def save_corrections_to_database(self) -> bool:
        """
        Save user corrections to database and learn from them for future improvements.

        Returns:
            bool: True if corrections were saved successfully, False otherwise
        """
        from datetime import datetime
        import traceback
        import sqlite3
        import streamlit as st

        # ============= VALIDATION CHECKS =============

        # Check 1: Learning database availability
        if not hasattr(self, 'learning_db') or not self.learning_db:
            st.error(
                "âŒ Learning database not available - corrections cannot be saved")
            if st.button("ðŸ”„ Reinitialize Database", key="reinit_db_from_save"):
                try:
                    self._init_learning_systems()
                    st.success("âœ… Database reinitialized - try saving again")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"âŒ Reinitialization failed: {e}")
            return False

        # Check 2: Document loaded
        if not hasattr(st.session_state, 'current_document_hash') or not st.session_state.current_document_hash:
            st.error("âŒ No document loaded - cannot save corrections")
            return False

        # Check 3: Extracted fields exist
        if not hasattr(st.session_state, 'extracted_fields') or not st.session_state.extracted_fields:
            st.error("âŒ No extracted fields found - nothing to save")
            return False

        # ============= DATABASE CONNECTION TEST =============

        try:
            # Test database connection before proceeding
            with sqlite3.connect(self.learning_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM corrections")
                existing_corrections = cursor.fetchone()[0]

            print(
                f"ðŸ” DEBUG: Database connection verified - {existing_corrections} existing corrections")

        except Exception as db_test_error:
            st.error(f"âŒ Database connection test failed: {db_test_error}")
            st.error("Please check database diagnostics in Settings")
            return False

        # ============= CORRECTION PROCESSING =============

        try:
            document_hash = st.session_state.current_document_hash
            pdf_name = st.session_state.get(
                'current_pdf_name', 'Unknown Document')
            extracted_text = st.session_state.get('extracted_text', '')

            print(
                f"ðŸ” DEBUG: Saving corrections for document {document_hash[:8]}...")
            print(f"ðŸ” DEBUG: PDF name: {pdf_name}")
            print(f"ðŸ” DEBUG: Text length: {len(extracted_text)} characters")

            corrections_saved = 0
            patterns_learned = 0
            failed_saves = 0
            failed_patterns = 0

            # Process each field correction
            for field_name, corrected_value in st.session_state.extracted_fields.items():
                if corrected_value and str(corrected_value).strip():
                    corrected_value_clean = str(corrected_value).strip()

                    try:
                        # ===== SAVE CORRECTION TO DATABASE =====

                        # Get original value for comparison
                        original_value = st.session_state.get(
                            'original_extracted_fields', {}).get(field_name, '')

                        success = self.learning_db.save_correction(
                            document_hash=document_hash,
                            pdf_name=pdf_name,
                            field_name=field_name,
                            corrected_value=corrected_value_clean,
                            confidence=0.95,  # User corrections are high confidence
                            original_value=original_value
                        )

                        if success:
                            corrections_saved += 1
                            print(
                                f"âœ… DEBUG: Saved correction for {field_name}: '{original_value}' â†’ '{corrected_value_clean}'")

                            # ===== PATTERN LEARNING =====

                            if hasattr(self, 'pattern_learner') and self.pattern_learner:
                                try:
                                    # Prepare context for pattern learning
                                    # Limit context size
                                    context_length = min(
                                        len(extracted_text), 2000)
                                    context = extracted_text[:context_length]

                                    # Learn from this correction
                                    self.pattern_learner.learn_from_feedback(
                                        field_type=field_name,
                                        correct_value=corrected_value_clean,
                                        context=context
                                    )
                                    patterns_learned += 1
                                    print(
                                        f"ðŸ§  DEBUG: Learned pattern for {field_name}")

                                except Exception as pattern_error:
                                    failed_patterns += 1
                                    print(
                                        f"âš ï¸ DEBUG: Pattern learning failed for {field_name}: {pattern_error}")
                                    # Don't fail the entire save operation
                            else:
                                print(
                                    f"âš ï¸ DEBUG: Pattern learner not available for {field_name}")
                        else:
                            failed_saves += 1
                            print(
                                f"âŒ DEBUG: Failed to save correction for {field_name}")

                    except Exception as field_error:
                        failed_saves += 1
                        print(
                            f"âŒ DEBUG: Error processing {field_name}: {field_error}")
                        continue

            # ============= RESULTS AND FEEDBACK =============

            if corrections_saved > 0:
                # Success message with details
                if patterns_learned > 0:
                    st.success(
                        f"âœ… Saved {corrections_saved} corrections and learned {patterns_learned} new patterns!")
                    st.info(
                        "ðŸ§  The system will use these patterns to improve future extractions")
                else:
                    st.success(
                        f"âœ… Successfully saved {corrections_saved} corrections to database!")

                # Show any failures
                if failed_saves > 0:
                    st.warning(f"âš ï¸ {failed_saves} corrections failed to save")

                if failed_patterns > 0:
                    st.info(
                        f"â„¹ï¸ Pattern learning failed for {failed_patterns} fields (corrections still saved)")

                # ===== UPDATE PERFORMANCE METRICS =====

                if 'performance_metrics' not in st.session_state:
                    st.session_state.performance_metrics = {
                        'total_documents': 0,
                        'average_accuracy': 0.85,
                        'total_corrections': 0,
                        'patterns_learned': 0,
                        'last_updated': None
                    }

                st.session_state.performance_metrics['total_corrections'] += corrections_saved
                st.session_state.performance_metrics['patterns_learned'] = st.session_state.performance_metrics.get(
                    'patterns_learned', 0) + patterns_learned
                st.session_state.performance_metrics['last_updated'] = datetime.now(
                ).isoformat()

                # ===== VERIFY SAVE OPERATION =====

                try:
                    # Verify corrections were actually saved
                    saved_corrections = self.learning_db.get_corrections_for_document(
                        document_hash)
                    verification_count = len(saved_corrections)

                    if verification_count >= corrections_saved:
                        print(
                            f"âœ… DEBUG: Verification successful - {verification_count} corrections in database")
                        st.success(
                            f"ðŸ” Verification: {verification_count} corrections confirmed in database")
                    else:
                        st.warning(
                            f"âš ï¸ Verification: Only {verification_count} corrections found in database")

                    # Show recent corrections
                    if saved_corrections:
                        with st.expander("ðŸ“‹ View Saved Corrections", expanded=False):
                            for field, value in saved_corrections.items():
                                st.write(f"**{field}:** {value}")

                except Exception as verify_error:
                    st.warning(
                        f"âš ï¸ Could not verify corrections were saved: {verify_error}")

                # ===== SUCCESS STATISTICS =====

                success_rate = (corrections_saved / (corrections_saved + failed_saves)
                                ) * 100 if (corrections_saved + failed_saves) > 0 else 0
                pattern_rate = (patterns_learned / corrections_saved) * \
                    100 if corrections_saved > 0 else 0

                print(
                    f"ðŸ“Š DEBUG: Success rate: {success_rate:.1f}%, Pattern learning rate: {pattern_rate:.1f}%")

                return True

            else:
                if failed_saves > 0:
                    st.error(f"âŒ All {failed_saves} correction saves failed")
                else:
                    st.warning(
                        "âš ï¸ No corrections to save - all fields are empty")
                return False

        except Exception as e:
            st.error(f"âŒ Unexpected error saving corrections: {e}")
            print(f"âŒ DEBUG: Exception details: {type(e).__name__}: {str(e)}")

            # Show detailed error for debugging
            with st.expander("ðŸ” Error Details (for debugging)", expanded=False):
                st.code(traceback.format_exc())

            return False

        finally:
            # ===== CLEANUP AND LOGGING =====

            print("ðŸ” DEBUG: save_corrections_to_database operation completed")

            # Log operation summary
            if 'corrections_saved' in locals():
                print(
                    f"ðŸ“Š DEBUG: Final summary - Saved: {corrections_saved}, Patterns: {patterns_learned}, Failed: {failed_saves}")

    def save_corrections(self, original_fields: Dict, corrected_fields: Dict):
        """Save user corrections for adaptive learning."""
        corrections = []

        for field_name, original_value in original_fields.items():
            corrected_value = corrected_fields.get(field_name, "")

            if original_value != corrected_value and corrected_value.strip():
                corrections.append({
                    'field_name': field_name,
                    'original_value': original_value,
                    'corrected_value': corrected_value,
                    'timestamp': datetime.now().isoformat(),
                    'document_hash': st.session_state.get('current_document_hash', ''),
                    'pdf_name': st.session_state.get('current_pdf_name', '')
                })

        # Save to corrections history
        if 'corrections_history' not in st.session_state:
            st.session_state.corrections_history = []

        st.session_state.corrections_history.extend(corrections)

        # Save to database if available
        if self.learning_db:
            for correction in corrections:
                self.learning_db.save_correction(
                    document_hash=correction['document_hash'],
                    pdf_name=correction['pdf_name'],
                    field_name=correction['field_name'],
                    corrected_value=correction['corrected_value']
                )

        if corrections:
            st.success(f"Saved {len(corrections)} corrections!")
            return True
        else:
            st.info("No corrections to save.")
            return False

    # Add the missing main execution block

    def apply_table_corrections(self, original_table: dict, corrected_df: pd.DataFrame, table_index: int):
        """Apply table corrections and trigger learning."""
        try:
            # Convert corrected DataFrame back to table format
            corrected_table = {
                'headers': corrected_df.columns.tolist(),
                'data': corrected_df.values.tolist(),
                'page': original_table.get('page', 1),
                'confidence': 1.0,  # Corrected data is 100% accurate
                'method': 'user_corrected'
            }

            # Save correction to learning database
            if hasattr(self, 'learning_db') and self.learning_db:
                self.learning_db.save_table_correction(
                    original_table,
                    corrected_table,
                    st.session_state.current_pdf_name,
                    original_table.get('page', 1)
                )

                # Extract and save learned patterns
                self._extract_table_learning_patterns(
                    original_table, corrected_table)

            # Update the table in session state
            st.session_state.extracted_tables[table_index] = corrected_table

            # Update performance metrics
            st.session_state.performance_metrics['total_corrections'] += 1

        except Exception as e:
            st.error(f"Error applying table corrections: {e}")

    def train_from_table_corrections(self, original_table: dict, corrected_df: pd.DataFrame, table_index: int):
        """Immediately train model from table corrections."""
        try:
            # Apply corrections first
            self.apply_table_corrections(
                original_table, corrected_df, table_index)

            # Trigger immediate retraining if extractor supports it
            from secure_pdf_extractor.extraction.table_extraction import StreamlitCompatibleDebugExtractor

            extractor = StreamlitCompatibleDebugExtractor()

            # Learn from the correction
            corrected_table = {
                'headers': corrected_df.columns.tolist(),
                'data': corrected_df.values.tolist()
            }

            extractor.learn_from_table_corrections(
                original_table, corrected_table)

            # Update extraction confidence
            self._update_extraction_confidence()

        except Exception as e:
            st.error(f"Error training from table corrections: {e}")

    def _extract_table_learning_patterns(self, original_table: dict, corrected_table: dict):
        """Extract learning patterns from table corrections."""
        try:
            if not hasattr(self, 'learning_db') or not self.learning_db:
                return

            # Learn from header corrections
            original_headers = original_table.get('headers', [])
            corrected_headers = corrected_table.get('headers', [])

            for i, (orig_header, corr_header) in enumerate(zip(original_headers, corrected_headers)):
                if orig_header != corr_header:
                    self.learning_db.save_learned_table_pattern(
                        'header_correction',
                        orig_header,
                        corr_header,
                        f'position_{i}',
                        0.2
                    )

            # Learn from data corrections
            original_data = original_table.get('data', [])
            corrected_data = corrected_table.get('data', [])

            for row_idx, (orig_row, corr_row) in enumerate(zip(original_data, corrected_data)):
                for col_idx, (orig_cell, corr_cell) in enumerate(zip(orig_row, corr_row)):
                    if str(orig_cell) != str(corr_cell):
                        self.learning_db.save_learned_table_pattern(
                            'data_correction',
                            str(orig_cell),
                            str(corr_cell),
                            f'row_{row_idx}_col_{col_idx}',
                            0.1
                        )

        except Exception as e:
            logger.warning(f"Error extracting table learning patterns: {e}")

    def _update_extraction_confidence(self):
        """Update overall extraction confidence based on corrections."""
        try:
            if hasattr(self, 'learning_db') and self.learning_db:
                stats = self.learning_db.get_table_correction_stats()

                # Update performance metrics
                if stats['total_corrections'] > 0:
                    improvement = stats['avg_confidence_improvement']
                    st.session_state.performance_metrics['average_accuracy'] = min(1.0,
                                                                                   st.session_state.performance_metrics['average_accuracy'] + improvement * 0.1)

        except Exception as e:
            logger.warning(f"Error updating extraction confidence: {e}")

    def render_performance_dashboard(self):
        """Render performance dashboard."""
        st.header("Performance Dashboard")

        metrics = st.session_state.performance_metrics

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Documents", metrics['total_documents'])

        with col2:
            st.metric("Average Accuracy", f"{metrics['average_accuracy']:.1%}")

        with col3:
            st.metric("Total Corrections", metrics['total_corrections'])

        with col4:
            accuracy_trend = "+2.3%" if metrics['total_corrections'] > 0 else "0%"
            st.metric("Accuracy Trend", accuracy_trend)

        # Confidence scores chart
        if st.session_state.confidence_scores:
            st.subheader("Field Confidence Scores")
            # Calculate stats for 4 fields only
            found_count = sum(1 for field in ['date', 'angebot', 'company_name', 'sender_address']
                              if st.session_state.extracted_fields.get(field, '').strip())
            total_count = 4  # Always 4 fields
            success_rate = (found_count / total_count) * 100

            # Display 4-field statistics
            st.subheader("4-Field Extraction Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fields Found", f"{found_count}/4")
            with col2:
                st.metric("Success Rate", f"{success_rate:.1f}%")

            field_display_names = {'date': 'Date',
                                   'angebot': 'Angebot Number',
                                   'company_name': 'Company Name',
                                   'sender_address': 'Company Address'
                                   }

            df = pd.DataFrame([
                {'Field': field_display_names.get(
                    field, field), 'Confidence': score}
                for field, score in st.session_state.confidence_scores.items()
                if field in ['date', 'angebot', 'company_name', 'sender_address']
            ])

    def render_learning_progress(self):
        """Enhanced learning progress interface with pattern learner integration and improved organization."""
        st.header("ðŸ§  Adaptive Learning Progress")

        # Safely get corrections history from session state
        corrections = st.session_state.get('corrections_history', [])

        # Get adaptive learning statistics with robust error handling
        adaptive_stats = None
        if hasattr(self, 'adaptive_extractor') and self.adaptive_extractor:
            try:
                adaptive_stats = self.adaptive_extractor.get_learning_statistics()
            except Exception as e:
                st.warning(f"Could not get adaptive learning stats: {e}")

        # Get fallback table learning statistics
        table_stats = {
            'total_table_corrections': 0,
            'average_confidence_improvement': 0.0,
            'documents_with_table_corrections': 0,
            'learned_pattern_count': 0
        }
        if hasattr(self, 'learning_db') and self.learning_db:
            try:
                table_stats = self.learning_db.get_table_correction_stats()
            except Exception as e:
                st.warning(f"Could not get table correction stats: {e}")

        # âœ… NEW: Get pattern learner statistics
        pattern_stats = {
            'total_patterns': 0,
            'patterns_by_type': [],
            'top_patterns': []
        }
        if hasattr(self, 'pattern_learner') and self.pattern_learner:
            try:
                pattern_stats = self.pattern_learner.get_pattern_statistics()
            except Exception as e:
                st.warning(f"Could not get pattern learner stats: {e}")

        # Calculate core metrics
        total_field_corrections = len(corrections)

        if adaptive_stats:
            total_table_corrections = adaptive_stats.get(
                'header_corrections', 0) + adaptive_stats.get('cell_corrections', 0)
            templates_learned = adaptive_stats.get('templates_learned', 0)
            patterns_learned = adaptive_stats.get('patterns_learned', 0)
            success_rate = adaptive_stats.get('average_success_rate', 0)
            extraction_time = adaptive_stats.get(
                'average_extraction_time_ms', 0)
        else:
            total_table_corrections = table_stats['total_table_corrections']
            templates_learned = 0
            patterns_learned = table_stats['learned_pattern_count']
            success_rate = 0
            extraction_time = 0

        # âœ… NEW: Add pattern learner patterns to total
        pattern_learner_patterns = pattern_stats['total_patterns']
        total_patterns_learned = patterns_learned + pattern_learner_patterns

        # Calculate recent corrections (last 7 days)
        recent_corrections = 0
        try:
            one_week_ago = datetime.now() - timedelta(days=7)
            recent_corrections = len([
                c for c in corrections
                if 'timestamp' in c and pd.to_datetime(c['timestamp'], errors='coerce') > one_week_ago
            ])
        except Exception:
            recent_corrections = 0

        # Display primary metrics
        st.subheader("ðŸ“Š Learning Statistics Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Field Corrections", total_field_corrections)
            st.metric("Table Corrections", total_table_corrections)

        with col2:
            st.metric("This Week (Fields)", recent_corrections)
            st.metric("Templates Learned", templates_learned)

        with col3:
            # Field improvement calculation
            field_improvement = min(
                total_field_corrections * 0.02, 0.15) if total_field_corrections > 0 else 0
            st.metric("Field Improvement", f"+{field_improvement:.1%}")

            # Success rate or table improvement
            if adaptive_stats:
                st.metric("Success Rate", f"{success_rate:.1%}")
            else:
                table_improvement = table_stats['average_confidence_improvement']
                st.metric("Table Improvement", f"+{table_improvement:.1%}")

        with col4:
            st.metric("Total Patterns", total_patterns_learned)

            # Progress to 90% target
            current_accuracy = st.session_state.get(
                'performance_metrics', {}).get('average_accuracy', 0.0)
            progress_to_90 = min(current_accuracy / 0.9,
                                 1.0) if current_accuracy > 0 else 0.0
            st.metric("Progress to 90%", f"{progress_to_90:.1%}")

        # âœ… NEW: Pattern Learning Statistics Section
        if hasattr(self, 'pattern_learner') and self.pattern_learner:
            st.subheader("ðŸ§  Pattern Learning Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Field Patterns", pattern_learner_patterns)
                field_types_count = len(pattern_stats['patterns_by_type'])
                st.metric("Field Types Learned", field_types_count)

            with col2:
                # Get patterns for each field type
                date_patterns = len(self.pattern_learner.get_learned_patterns(
                    'date')) if pattern_learner_patterns > 0 else 0
                angebot_patterns = len(self.pattern_learner.get_learned_patterns(
                    'angebot')) if pattern_learner_patterns > 0 else 0
                st.metric("Date Patterns", date_patterns)
                st.metric("Angebot Patterns", angebot_patterns)

            with col3:
                company_patterns = len(self.pattern_learner.get_learned_patterns(
                    'company_name')) if pattern_learner_patterns > 0 else 0
                address_patterns = len(self.pattern_learner.get_learned_patterns(
                    'sender_address')) if pattern_learner_patterns > 0 else 0
                st.metric("Company Patterns", company_patterns)
                st.metric("Address Patterns", address_patterns)

            with col4:
                # Calculate average confidence from top patterns
                avg_confidence = 0.0
                if pattern_stats['top_patterns']:
                    confidences = [p[1]
                                   for p in pattern_stats['top_patterns'] if len(p) > 1]
                    avg_confidence = sum(confidences) / \
                        len(confidences) if confidences else 0.0

                st.metric("Avg Pattern Confidence", f"{avg_confidence:.1%}")

                # Pattern learning efficiency
                if total_field_corrections > 0 and pattern_learner_patterns > 0:
                    efficiency = (pattern_learner_patterns /
                                  total_field_corrections) * 100
                    st.metric("Learning Efficiency", f"{efficiency:.0f}%")
                else:
                    st.metric("Learning Efficiency", "0%")

        # Advanced metrics (only if adaptive stats available)
        if adaptive_stats:
            st.subheader("ðŸ”¬ Advanced Learning Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Header Corrections", adaptive_stats.get(
                    'header_corrections', 0))
                st.metric("Cell Corrections", adaptive_stats.get(
                    'cell_corrections', 0))

            with col2:
                st.metric("Field Corrections", adaptive_stats.get(
                    'field_corrections', 0))
                st.metric("Total Corrections", adaptive_stats.get(
                    'total_corrections', 0))

            with col3:
                st.metric("Extraction Time", f"{extraction_time:.0f}ms")

                # Calculate documents processed
                docs_processed = adaptive_stats.get('templates_learned', 0)
                if adaptive_stats.get('documents_with_table_corrections'):
                    docs_processed = max(
                        docs_processed, adaptive_stats['documents_with_table_corrections'])
                st.metric("Documents Processed", docs_processed)

            with col4:
                # Pattern type breakdown
                pattern_types = adaptive_stats.get('pattern_types', {})
                if pattern_types:
                    st.write("**Pattern Types:**")
                    for pattern_type, count in pattern_types.items():
                        clean_name = pattern_type.replace('_', ' ').title()
                        st.write(f"â€¢ {clean_name}: {count}")
                else:
                    st.write("**No patterns learned yet**")

        # German table extraction status
        st.subheader("ðŸ‡©ðŸ‡ª German Document Processing Enhancement")
        c1, c2 = st.columns(2)

        with c1:
            st.info("**Target Accuracy: 90%** for German business documents")
            st.write("- Adaptive field pattern learning")
            st.write("- Borderless table detection")
            st.write("- Advanced German language patterns")
            st.write("- Automatic learning from corrections")

            # Enhanced system status indicator
            systems_active = 0
            total_systems = 3

            if adaptive_stats:
                st.success("âœ… Adaptive table learning: Active")
                systems_active += 1
            elif hasattr(self, 'learning_db') and self.learning_db:
                st.warning("âš ï¸ Basic table learning: Active")
                systems_active += 0.5
            else:
                st.error("âŒ Table learning: Inactive")

            if hasattr(self, 'pattern_learner') and self.pattern_learner:
                st.success("âœ… Pattern learning: Active")
                systems_active += 1
            else:
                st.error("âŒ Pattern learning: Inactive")

            if hasattr(self, 'learning_db') and self.learning_db:
                st.success("âœ… Learning database: Connected")
                systems_active += 1
            else:
                st.error("âŒ Learning database: Disconnected")

            # Overall system health
            health_percentage = (systems_active / total_systems) * 100
            if health_percentage >= 90:
                st.success(f"ðŸŸ¢ System Health: {health_percentage:.0f}%")
            elif health_percentage >= 60:
                st.warning(f"ðŸŸ¡ System Health: {health_percentage:.0f}%")
            else:
                st.error(f"ðŸ”´ System Health: {health_percentage:.0f}%")

        with c2:
            # Use appropriate stats based on available system
            corrections_count = total_table_corrections if adaptive_stats else table_stats[
                'total_table_corrections']
            confidence_boost = table_stats['average_confidence_improvement']
            files_improved = adaptive_stats.get(
                'templates_learned', 0) if adaptive_stats else table_stats['documents_with_table_corrections']

            if corrections_count > 0 or pattern_learner_patterns > 0:
                st.success(
                    f"**{total_patterns_learned} total patterns learned**")
                if pattern_learner_patterns > 0:
                    st.write(f"- Field patterns: {pattern_learner_patterns}")
                if patterns_learned > 0:
                    st.write(f"- Table patterns: {patterns_learned}")
                st.write(
                    f"- Average confidence boost: +{confidence_boost:.1%}")
                st.write(
                    f"- Documents improved: {max(files_improved, total_field_corrections)}")

                # Enhanced learning efficiency indicator
                total_corrections_all = total_field_corrections + corrections_count
                if total_corrections_all > 0 and total_patterns_learned > 0:
                    efficiency = (total_patterns_learned /
                                  total_corrections_all) * 100
                    st.write(
                        f"- Overall learning efficiency: {efficiency:.0f}%")
            else:
                st.info(
                    "No corrections made yet - start correcting fields and tables to improve accuracy!")

        # âœ… ENHANCED: Pattern Learning Display
        if hasattr(self, 'pattern_learner') and self.pattern_learner and pattern_learner_patterns > 0:
            st.subheader("ðŸ” Learned Field Patterns")

            # Create tabs for different field types
            field_types = ['date', 'angebot', 'company_name', 'sender_address']
            tabs = st.tabs(
                [f"ðŸ“… Date", f"ðŸ“„ Angebot", f"ðŸ¢ Company", f"ðŸ“ Address"])

            for i, (tab, field_type) in enumerate(zip(tabs, field_types)):
                with tab:
                    try:
                        patterns = self.pattern_learner.get_learned_patterns(
                            field_type)
                        if patterns:
                            pattern_data = []
                            for pattern in patterns[:5]:  # Show top 5 patterns
                                pattern_data.append({
                                    'Pattern': pattern['pattern'][:50] + '...' if len(pattern['pattern']) > 50 else pattern['pattern'],
                                    'Confidence': f"{pattern['confidence']:.2f}",
                                    'Success Count': pattern['success_count'],
                                    'Total Usage': pattern['total_count']
                                })

                            if pattern_data:
                                pattern_df = pd.DataFrame(pattern_data)
                                st.dataframe(
                                    pattern_df, use_container_width=True)
                            else:
                                st.info(
                                    f"No patterns learned for {field_type.replace('_', ' ')} yet.")
                        else:
                            st.info(
                                f"No patterns learned for {field_type.replace('_', ' ')} yet.")
                    except Exception as e:
                        st.warning(
                            f"Could not load {field_type} patterns: {e}")

        # Recent field corrections history
        if corrections:
            st.subheader("ðŸ•˜ Recent Field Corrections")
            try:
                corrections_df = pd.DataFrame(corrections)

                # Convert timestamps safely
                if 'timestamp' in corrections_df.columns:
                    corrections_df['timestamp'] = pd.to_datetime(
                        corrections_df['timestamp'], errors='coerce')
                    corrections_df = corrections_df.sort_values(
                        'timestamp', ascending=False)
                else:
                    corrections_df['timestamp'] = pd.NaT

                # Display recent corrections
                for _, correction in corrections_df.head(5).iterrows():
                    field_name = correction.get('field_name', 'Unknown field')
                    timestamp = correction['timestamp']
                    timestamp_str = timestamp.strftime(
                        '%Y-%m-%d %H:%M') if pd.notna(timestamp) else "Unknown time"

                    with st.expander(f"{field_name} - {timestamp_str}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("**Original:**")
                            st.code(correction.get('original_value', ''))
                        with c2:
                            st.write("**Corrected:**")
                            st.code(correction.get('corrected_value', ''))
            except Exception as e:
                st.warning(f"Error displaying correction history: {e}")

        # Legacy patterns display (fallback)
        elif hasattr(self, 'learning_db') and self.learning_db and table_stats['learned_pattern_count'] > 0:
            st.subheader("ðŸ“Š Learned Table Patterns")
            try:
                patterns = self.learning_db.get_learned_table_patterns()
                if patterns:
                    pattern_df = pd.DataFrame(patterns[:10])
                    display_columns = [
                        'pattern_type', 'original_value', 'corrected_value', 'usage_count']
                    available_columns = [
                        col for col in display_columns if col in pattern_df.columns]
                    if available_columns:
                        st.dataframe(
                            pattern_df[available_columns], use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load table patterns: {e}")

        # âœ… ENHANCED: Intelligent recommendations
        st.subheader("ðŸ’¡ Learning Recommendations")
        total_corrections = total_field_corrections + total_table_corrections

        if not hasattr(self, 'pattern_learner') and not hasattr(self, 'learning_db'):
            st.error(
                "ðŸš¨ **Critical**: No learning systems detected. Initialize pattern learner and learning database!")
        elif not hasattr(self, 'pattern_learner'):
            st.warning(
                "âš ï¸ **Missing**: Pattern learner not initialized. Field extraction won't improve over time.")
        elif total_corrections < 5:
            st.info(
                "ðŸš€ **Get Started**: Process more documents and make corrections to improve accuracy")
            st.write(
                "ðŸ’¡ *Tip: Start with 5-10 corrections to see meaningful learning patterns*")
        elif pattern_learner_patterns < 3:
            st.info(
                "ðŸŽ¯ **Pattern Focus**: Make more field corrections to build pattern library")
            st.write(
                "ðŸ’¡ *Tip: Each field type needs at least 2-3 corrections to learn effective patterns*")
        elif adaptive_stats and adaptive_stats.get('templates_learned', 0) < 3:
            st.info(
                "ðŸ“‹ **Template Focus**: Process different document types to improve template recognition")
        elif total_table_corrections < 3:
            st.info(
                "ðŸ“Š **Table Focus**: Try correcting table data to boost German document accuracy")
            st.write(
                "ðŸ’¡ *Tip: Table corrections have higher impact on overall accuracy*")
        elif current_accuracy < 0.8:
            st.warning(
                "ðŸŽ¯ **Accuracy Focus**: Continue making corrections to reach 90% target accuracy")
            remaining = int((0.9 - current_accuracy) * 100)
            st.write(
                f"ðŸ’¡ *Need approximately {remaining} more percentage points to reach target*")
        elif success_rate > 0 and success_rate >= 0.85:
            st.success(
                "ðŸŒŸ **Excellent**: Your system is learning well and approaching 90% accuracy!")
            st.write(
                "ðŸ’¡ *Consider expanding to new document types to test generalization*")
        else:
            st.success(
                "ðŸ‘ **Good Progress**: System is learning steadily. Keep making corrections!")

        # System health indicators
        st.subheader("ðŸ”§ System Health")
        health_col1, health_col2 = st.columns(2)

        with health_col1:
            # Database health assessment
            if adaptive_stats:
                total_corrections_count = adaptive_stats.get(
                    'total_corrections', 0)
                if total_corrections_count > 10:
                    db_health = "âœ… Excellent"
                elif total_corrections_count > 5:
                    db_health = "ðŸŸ¡ Good"
                else:
                    db_health = "ðŸŸ  Building"

                st.write(f"**Database Health**: {db_health}")

                # Learning rate calculation
                learning_rate = (adaptive_stats.get(
                    'patterns_learned', 0) / max(adaptive_stats.get('total_corrections', 1), 1))
                st.write(
                    f"**Learning Rate**: {learning_rate:.2f} patterns/correction")
            else:
                st.write("**Database Health**: ðŸŸ¡ Basic")

            # Pattern learner health
            if hasattr(self, 'pattern_learner') and pattern_learner_patterns > 0:
                pattern_health = "âœ… Active" if pattern_learner_patterns > 5 else "ðŸŸ¡ Learning"
                st.write(f"**Pattern Learning**: {pattern_health}")

        with health_col2:
            # Performance indicators
            if extraction_time > 0:
                if extraction_time < 2000:
                    perf_indicator = "âš¡ Fast"
                elif extraction_time < 5000:
                    perf_indicator = "ðŸŸ¡ Normal"
                else:
                    perf_indicator = "ðŸŸ  Slow"
                st.write(f"**Performance**: {perf_indicator}")

            # Learning trend assessment
            if total_corrections > 0:
                if success_rate > 0.7 or total_patterns_learned > 10:
                    trend = "ðŸ“ˆ Improving"
                elif total_patterns_learned > 0:
                    trend = "ðŸ”„ Steady"
                else:
                    trend = "ðŸš€ Starting"
                st.write(f"**Learning Trend**: {trend}")

            # Overall system readiness
            if pattern_learner_patterns > 3 and total_corrections > 5:
                st.success("**System Status**: ðŸŸ¢ Production Ready")
            elif pattern_learner_patterns > 0 or total_corrections > 0:
                st.info("**System Status**: ðŸŸ¡ Learning Mode")
            else:
                st.warning("**System Status**: ðŸŸ  Setup Required")

    def render_training_data(self):
        """Render training data upload and management interface."""
        st.header("Training Data Management")

        # Training data upload section
        st.subheader(" Upload Training Data")

        # Show supported formats
        with st.expander("â‚¬Â¹ Supported File Formats"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Documents:**")
                st.write("PDF Documents")
                st.write("Word Documents (DOCX, DOC)")
                st.write("Plain Text Files (TXT)")

            with col2:
                st.markdown("**Images:**")
                st.write("PNG, JPG, JPEG")
                st.write("TIFF, BMP")
                st.write("(OCR text extraction)")

            with col3:
                st.markdown("**Data Files:**")
                st.write("JSON, XML")
                st.write("CSV Spreadsheets")
                st.write("Excel Files (XLSX, XLS)")

        uploaded_files = st.file_uploader(
            "Upload Training Documents (Any Format)",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'docx',
                  'doc', 'txt', 'json', 'xml', 'csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload documents in any supported format to expand the training dataset",
            key="training_documents_upload"
        )

        if uploaded_files:
            st.success(f"â‚¬Â¦ {len(uploaded_files)} files uploaded")

            if st.button(" Process Training Files", key="process_training_files", type="primary"):
                self.process_training_files(uploaded_files)

        # Training and Dataset Management Actions (all in main layout for visibility)
        st.subheader("Training & Dataset Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Training Actions**")
            if st.button(" Download Datasets",  key="download_datasets", help="Download public PDF datasets for training"):
                self.download_training_datasets()

            if st.button("  Train Models", key="train_models", help="Train the extraction models using available data"):
                self.train_models()

        with col2:
            st.markdown("Dataset Management")
            if st.button("Clear User Uploads", key="clear_user_uploads", help="Delete your uploaded training files"):
                self.clear_user_uploads()

            if st.button(" Delete All Datasets", key="delete_all_datasets", type="secondary", help="Delete ALL datasets and models (WARNING: Cannot be undone!)"):
                self.clear_all_datasets()

        with col3:
            st.markdown("Learning Management")
            if st.button("Clear Learning History", key="clear_learning_history", help="Delete corrections and learned patterns (keeps datasets)"):
                self.clear_learning_data()

            if st.button("Clear Processed Docs", key="clear_processed_docs", help="Delete user processed training documents"):
                self.clear_processed_documents()

            # Add a simpler, more reliable delete option
            if st.button(" Quick Delete All",  key="quick_delete_all", type="secondary", help="Immediately delete all training data"):
                self.quick_delete_all()

        # Advanced delete with confirmation
        with st.expander("Advanced Delete Options"):
            if st.button(" Delete All Datasets (Advanced)", type="secondary", help="Delete ALL datasets and models with confirmation"):
                self.clear_all_datasets()

        # Current training data status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Training Data Status")
        with col2:
            if st.button(" Refresh Status", key="refresh_status_btn", help="Refresh training data status"):
                st.rerun()

        # Check current datasets - always check from actual filesystem, not cached data
        try:
            dataset_summary_path = Path("data/dataset_summary.json")
            data_dir = Path("data")

            # Count actual files from user uploads and training directories ONLY
            total_files = 0
            german_files = 0
            english_files = 0

            if data_dir.exists():
                # Count files in user_uploads directory
                user_uploads_dir = data_dir / "user_uploads"
                if user_uploads_dir.exists():
                    files = list(user_uploads_dir.rglob("*"))
                    upload_count = len([f for f in files if f.is_file() and f.suffix.lower() in [
                                       '.pdf', '.png', '.jpg', '.jpeg', '.docx', '.txt', '.json', '.xml', '.csv', '.xlsx']])
                    total_files += upload_count
                    english_files += upload_count  # Default to English for user uploads

                # Count files in user_training directory (processed documents)
                user_training_dir = data_dir / "user_training"
                if user_training_dir.exists():
                    training_files = list(user_training_dir.glob("*.json"))
                    total_files += len(training_files)

                    # Check each training file for language
                    for training_file in training_files:
                        try:
                            with open(training_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            # Check if it's German based on content or filename
                            if 'de' in data.get('language', '').lower() or 'german' in training_file.name.lower():
                                german_files += 1
                            else:
                                english_files += 1
                        except:
                            english_files += 1  # Default to English if can't read

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Files", total_files)

            with col2:
                st.metric("German Documents", german_files)

            with col3:
                st.metric("English Documents", english_files)

            # Don't show old dataset info from summary file - only show if there are actual current files
            if total_files == 0:
                st.info(
                    "â‚¬Â¹ No training datasets found. Upload training files to get started.")

                if st.button(" Start Training", key="start_training", type="primary"):
                    st.info(
                        "Use the upload section above to add training documents.")

        except Exception as e:
            st.error(f" Error loading training data status: {e}")
            # Reset to safe defaults
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", 0)
            with col2:
                st.metric("German Documents", 0)
            with col3:
                st.metric("English Documents", 0)

        # User processed documents
        st.subheader("Processed Documents")

        try:
            user_training_path = Path("data/user_training")
            if user_training_path.exists():
                training_files = list(user_training_path.glob("*.json"))

                if training_files:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Processed Documents", len(training_files))

                    with col2:
                        # Get latest training file
                        latest_file = max(
                            training_files, key=lambda x: x.stat().st_mtime)
                        last_trained = datetime.fromtimestamp(
                            latest_file.stat().st_mtime)
                        st.metric("Last Training",
                                  last_trained.strftime("%Y-%m-%d %H:%M"))

                    # Show recent documents
                    with st.expander("â‚¬Â¹ Recent Training Documents"):
                        recent_files = sorted(
                            training_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]

                        for file_path in recent_files:
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)

                                timestamp = datetime.fromisoformat(
                                    data['timestamp']).strftime("%Y-%m-%d %H:%M")
                                st.write(
                                    f" **{data['pdf_name']}** - {timestamp}")

                                # Show extracted fields
                                fields_preview = ", ".join([f"{k}: {v[:20]}..." if len(str(v)) > 20 else f"{k}: {v}"
                                                            for k, v in data['fields'].items() if v])
                                st.caption(fields_preview)

                            except Exception as e:
                                st.write(
                                    f" Error reading {file_path.name}: {e}")
                else:
                    st.info(
                        "No processed documents yet. Upload and process PDFs to start training.")
            else:
                st.info("No training data directory found.")

        except Exception as e:
            st.error(f"Error loading processed documents: {e}")

        # Model training status
        st.subheader("Model Training Status")

        try:
            training_results_path = Path("data/models/training_results.json")
            if training_results_path.exists():
                with open(training_results_path) as f:
                    results = json.load(f)

                col1, col2, col3 = st.columns(3)

                with col1:
                    accuracy = results.get("model_performance", {}).get(
                        "overall_accuracy", 0)
                    st.metric("Model Accuracy", f"{accuracy:.1%}")

                with col2:
                    patterns = results.get("model_performance", {}).get(
                        "pattern_training", {}).get("patterns_created", 0)
                    st.metric("Patterns Created", patterns)

                with col3:
                    docs = results.get("model_performance", {}).get(
                        "training_data_stats", {}).get("total_documents", 0)
                    st.metric("Training Documents", docs)

                # Training details
                with st.expander("Ã‹â€  Training Details"):
                    training_stats = results.get("model_performance", {}).get(
                        "training_data_stats", {})
                    pattern_stats = results.get(
                        "model_performance", {}).get("pattern_training", {})
                    eval_metrics = results.get("model_performance", {}).get(
                        "evaluation_metrics", {})

                    st.json({
                        "Training Data": training_stats,
                        "Pattern Training": pattern_stats,
                        "Evaluation Metrics": eval_metrics
                    })

            else:
                st.info(
                    "No model training results found. Train models to see performance metrics.")

                if st.button("  Start Model Training", key="start_model_training", type="primary"):
                    self.train_models()

        except Exception as e:
            st.error(f" Error loading model training status: {e}")

    def process_training_files(self, uploaded_files):
        """Process uploaded training files (multiple formats)."""
        with st.spinner("Processing uploaded training files..."):
            try:
                # Import multi-format extractor
                from secure_pdf_extractor.extraction.multi_extraction import MultiFormatExtractor, detect_file_type

                # Create training data directories
                uploads_dir = Path("data/user_uploads")
                training_dir = Path("data/user_training")
                uploads_dir.mkdir(parents=True, exist_ok=True)
                training_dir.mkdir(parents=True, exist_ok=True)

                processed_count = 0
                file_types = {}
                errors = []
                training_entries_created = 0

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        file_type = detect_file_type(uploaded_file.name)
                        status_text.text(
                            f"Processing {uploaded_file.name} ({file_type.upper()})...")

                        # Save original file to uploads
                        file_path = uploads_dir / uploaded_file.name
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())

                        # Extract text and data for training
                        try:
                            extractor = MultiFormatExtractor()
                            extracted_text = extractor.extract_text(file_path)

                            if extracted_text and len(extracted_text.strip()) > 10:
                                # Create training entry
                                training_entry = {
                                    "pdf_name": uploaded_file.name,
                                    "timestamp": datetime.now().isoformat(),
                                    # Limit text size
                                    "extracted_text": extracted_text[:2000],
                                    "fields": {},  # Will be filled when user makes corrections
                                    "language": "en",  # Default, will be detected later
                                    "source": "bulk_upload",
                                    "file_type": file_type
                                }

                                # Save training entry
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                safe_name = uploaded_file.name.replace(
                                    '.', '_').replace(' ', '_')
                                training_filename = f"training_{timestamp}_{safe_name}.json"
                                training_filepath = training_dir / training_filename

                                with open(training_filepath, 'w', encoding='utf-8') as f:
                                    json.dump(training_entry, f,
                                              indent=2, ensure_ascii=False)

                                training_entries_created += 1

                        except Exception as extract_error:
                            # Still count as processed even if extraction failed
                            errors.append(
                                f"{uploaded_file.name}: Could not extract text - {str(extract_error)}")

                        # Track file types
                        file_types[file_type] = file_types.get(
                            file_type, 0) + 1
                        processed_count += 1

                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))

                    except Exception as e:
                        errors.append(f"{uploaded_file.name}: {str(e)}")

                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()

                # Show results
                if processed_count > 0:
                    st.success(
                        f"â‚¬Â¦ Processed {processed_count} training files")
                    if training_entries_created > 0:
                        st.success(
                            f"â‚¬Â¹ Created {training_entries_created} training entries")

                    # Show file type breakdown
                    with st.expander("File Types Processed"):
                        for file_type, count in file_types.items():
                            st.write(f"**{file_type.upper()}**: {count} files")

                    st.info(
                        "Files saved and processed. Training data status will update automatically.")

                    # Force refresh of training data status
                    time.sleep(1)
                    st.rerun()

                # Show errors if any
                if errors:
                    st.warning("ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Â¸Â Some files had processing errors:")
                    for error in errors:
                        st.write(f"{error}")

            except Exception as e:
                st.error(f" Error processing training files: {e}")

    def download_training_datasets(self):
        """Download training datasets."""
        with st.spinner("Downloading training datasets..."):
            try:
                # Run dataset downloader
                import subprocess
                import sys

                result = subprocess.run([
                    sys.executable, "scripts/download_datasets.py"
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    st.success("â‚¬Â¦ Datasets downloaded successfully!")
                    st.code(result.stdout)
                    st.rerun()  # Refresh to show new data
                else:
                    st.error(" Dataset download failed")
                    st.code(result.stderr)

            except Exception as e:
                st.error(f" Error downloading datasets: {e}")

    def train_models(self):
        """Train models on available data."""
        with st.spinner("Training models on available data..."):
            try:
                # Run model trainer
                import subprocess
                import sys

                result = subprocess.run([
                    sys.executable, "scripts/train_models.py"
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    st.success("â‚¬Â¦ Model training completed!")
                    st.code(result.stdout)

                    # Update performance metrics
                    self.update_performance_from_training()
                    st.rerun()  # Refresh to show new metrics
                else:
                    st.error(" Model training failed")
                    st.code(result.stderr)

            except Exception as e:
                st.error(f" Error training models: {e}")

    def update_performance_from_training(self):
        """Update performance metrics from training results."""
        try:
            training_results_path = Path("data/models/training_results.json")
            if training_results_path.exists():
                with open(training_results_path) as f:
                    results = json.load(f)

                # Update session state metrics
                accuracy = results.get("model_performance", {}).get(
                    "overall_accuracy", 0)
                docs = results.get("model_performance", {}).get(
                    "training_data_stats", {}).get("total_documents", 0)

                st.session_state.performance_metrics.update({
                    "average_accuracy": accuracy,
                    "total_documents": docs
                })

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def clear_user_uploads(self):
        """Clear user uploaded training files."""
        try:
            import shutil

            user_uploads_dir = Path("data/user_uploads")

            if user_uploads_dir.exists():
                # Count files before deletion
                file_count = len(list(user_uploads_dir.glob("*.pdf")))

                if file_count == 0:
                    st.info("No user uploaded files to delete.")
                    return

                # Show confirmation dialog
                st.warning(
                    f"This will delete {file_count} uploaded PDF files. Are you sure?")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Yes, Delete Files", type="primary"):
                        # Delete all files in user uploads
                        shutil.rmtree(user_uploads_dir)
                        user_uploads_dir.mkdir(parents=True, exist_ok=True)

                        st.success(
                            f"Successfully deleted {file_count} user uploaded files!")
                        st.rerun()

                with col2:
                    if st.button("Cancel"):
                        st.info("Deletion cancelled.")
                        st.rerun()

                with col3:
                    st.write("")  # Spacer
            else:
                st.info("No user uploads directory found.")

        except Exception as e:
            st.error(f"Error clearing user uploads: {e}")

    def clear_all_datasets(self):
        """Clear ALL datasets, models, and training data - with confirmation."""
        # Initialize confirmation state if it doesn't exist
        if 'delete_confirmation_text' not in st.session_state:
            st.session_state.delete_confirmation_text = ""

        st.error("ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Â¸Â WARNING: This will delete ALL training data!")
        st.write("This includes:")
        st.write("All uploaded training files")
        st.write("All processed documents")
        st.write("All learning history and patterns")
        st.write("All downloaded datasets")
        st.write("All models and training results")

        st.warning("ÃƒÂ°Ã…Â¸Ã…Â¡Â¨ This action CANNOT be undone!")

        # Confirmation with typing
        st.write("Type **DELETE ALL** to confirm:")
        confirmation = st.text_input(
            "Confirmation", key="delete_confirmation_advanced")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("ÃƒÂ°Ã…Â¸â‚¬â€â‚¬ËœÃƒÂ¯Â¸Â PERMANENTLY DELETE ALL", type="primary", disabled=(confirmation != "DELETE ALL")):
                if confirmation == "DELETE ALL":
                    with st.spinner("Deleting all datasets and training data..."):
                        try:
                            import shutil

                            # Delete the entire data directory
                            data_dir = Path("data")
                            if data_dir.exists():
                                shutil.rmtree(data_dir)
                                st.success("â‚¬Â¦ Deleted data directory")

                            # Delete any root-level database files
                            root_files = [
                                "learning.db", "trained_patterns.db", "dataset_summary.json"]
                            for file_name in root_files:
                                file_path = Path(file_name)
                                if file_path.exists():
                                    file_path.unlink()
                                    st.success(f"â‚¬Â¦ Deleted {file_name}")

                            # Recreate clean directory structure
                            data_dir.mkdir(exist_ok=True)
                            (data_dir / "user_uploads").mkdir(exist_ok=True)
                            (data_dir / "user_training").mkdir(exist_ok=True)
                            (data_dir / "models").mkdir(exist_ok=True)
                            st.success(
                                "â‚¬Â¦ Recreated clean directory structure")

                            # Create empty learning database
                            learning_db_path = data_dir / "learning.db"
                            learning_db_path.touch()
                            st.success("â‚¬Â¦ Created new learning database")

                            # Clear ALL session state except current form
                            keys_to_delete = [k for k in st.session_state.keys() if k not in [
                                "delete_confirmation_advanced"]]
                            for key in keys_to_delete:
                                del st.session_state[key]

                            st.success("â‚¬Â¦ Cleared all session state")
                            st.success(
                                "ÃƒÂ°Ã…Â¸Ã…Â½â‚¬Â° ALL data has been permanently deleted!")
                            st.info(
                                "â‚¬Â¹ You can now start fresh by uploading new training documents.")

                            # Force immediate refresh
                            time.sleep(2)
                            st.rerun()

                        except Exception as e:
                            st.error(f" Error during deletion: {e}")
                            st.info(
                                "You may need to manually delete files and restart the application.")
                else:
                    st.error("Please type 'DELETE ALL' exactly to confirm.")

        with col2:
            if st.button(" Cancel"):
                # Clear the confirmation field
                st.session_state.delete_confirmation_text = ""
                st.info("Deletion cancelled.")
                st.rerun()

    def quick_delete_all(self):
        """Quickly delete all training data without confirmation."""
        st.warning("ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Â¸Â This will delete ALL training data immediately!")
        st.write("This includes:")
        st.write("All uploaded training files")
        st.write("All processed documents")
        st.write("All learning history and patterns")
        st.write("All downloaded datasets")
        st.write("All models and training results")

        st.warning("ÃƒÂ°Ã…Â¸Ã…Â¡Â¨ This action CANNOT be undone!")

        if st.button("ÃƒÂ°Ã…Â¸â‚¬â€â‚¬ËœÃƒÂ¯Â¸Â QUICKLY DELETE ALL", type="primary"):
            with st.spinner("Deleting all datasets and training data..."):
                try:
                    import shutil

                    # Delete the entire data directory
                    data_dir = Path("data")
                    if data_dir.exists():
                        shutil.rmtree(data_dir)
                        st.success("â‚¬Â¦ Deleted data directory")

                    # Delete any root-level database files
                    root_files = ["learning.db",
                                  "trained_patterns.db", "dataset_summary.json"]
                    for file_name in root_files:
                        file_path = Path(file_name)
                        if file_path.exists():
                            file_path.unlink()
                            st.success(f"â‚¬Â¦ Deleted {file_name}")

                    # Recreate clean directory structure
                    data_dir.mkdir(exist_ok=True)
                    (data_dir / "user_uploads").mkdir(exist_ok=True)
                    (data_dir / "user_training").mkdir(exist_ok=True)
                    (data_dir / "models").mkdir(exist_ok=True)
                    st.success("â‚¬Â¦ Recreated clean directory structure")

                    # Create empty learning database
                    learning_db_path = data_dir / "learning.db"
                    learning_db_path.touch()
                    st.success("â‚¬Â¦ Created new learning database")

                    # Clear ALL session state
                    keys_to_delete = list(st.session_state.keys())
                    for key in keys_to_delete:
                        if key != "delete_confirmation_final":  # Keep confirmation field
                            del st.session_state[key]

                    st.success("â‚¬Â¦ Cleared all session state")
                    st.success(
                        "ÃƒÂ°Ã…Â¸Ã…Â½â‚¬Â° ALL data has been permanently deleted!")
                    st.info(
                        "â‚¬Â¹ You can now start fresh by uploading new training documents.")

                    # Force immediate refresh
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f" Error during deletion: {e}")
                    st.info(
                        "You may need to manually delete files and restart the application.")

    def clear_learning_data(self):
        """Clear learning history and patterns but keep datasets."""
        try:
            learning_db_path = Path("data/learning.db")
            corrections_count = len(st.session_state.corrections_history)

            st.warning(
                f"This will delete {corrections_count} corrections and all learned patterns.")
            st.info("Your datasets and models will remain intact.")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Yes, Clear Learning Data", type="primary"):
                    # Delete learning database
                    if learning_db_path.exists():
                        learning_db_path.unlink()

                    # Reset session state learning data
                    st.session_state.corrections_history = []
                    st.session_state.performance_metrics['total_corrections'] = 0

                    st.success("Learning history and patterns cleared!")
                    st.info(
                        "The system will start learning fresh from new corrections.")
                    st.rerun()

            with col2:
                if st.button("Cancel"):
                    st.info("Clearing cancelled.")
                    st.rerun()

        except Exception as e:
            st.error(f"Error clearing learning data: {e}")

    def clear_processed_documents(self):
        """Clear user processed training documents."""
        try:
            user_training_path = Path("data/user_training")

            if user_training_path.exists():
                training_files = list(user_training_path.glob("*.json"))

                if not training_files:
                    st.info("No processed documents to delete.")
                    return

                st.warning(
                    f"This will delete {len(training_files)} processed training documents.")
                st.info("Your corrections and learned patterns will remain intact.")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Yes, Delete Processed Docs", type="primary"):
                        import shutil
                        shutil.rmtree(user_training_path)
                        user_training_path.mkdir(parents=True, exist_ok=True)

                        st.success(
                            f"Successfully deleted {len(training_files)} processed documents!")
                        st.rerun()

                with col2:
                    if st.button("Cancel"):
                        st.info("Deletion cancelled.")
                        st.rerun()
            else:
                st.info("No processed documents directory found.")

        except Exception as e:
            st.error(f"Error clearing processed documents: {e}")

    def train_from_current_document(self, original_fields: Dict, corrected_fields: Dict, auto_mode: bool = False):
        """Train the model using the current document's data."""
        try:
            if not auto_mode:
                st.info("  Training model with document data...")

            # Use corrected fields if available, otherwise use original
            training_data = {}
            for field_name, original_value in original_fields.items():
                corrected_value = corrected_fields.get(
                    field_name, original_value)
                training_data[field_name] = corrected_value

            # Get document context
            extracted_text = st.session_state.get('extracted_text', '')
            pdf_name = st.session_state.get('current_pdf_name', 'unknown')

            # Save training document
            self._save_training_document(
                pdf_name, extracted_text, training_data)

            # Learn patterns from the data
            self._learn_patterns_from_document(training_data, extracted_text)

            if not auto_mode:
                st.success("â‚¬Â¦ Model trained with document data!")
                st.info("The system has learned new patterns from this document.")
            else:
                st.success("â‚¬Â¦ Document added to training data!")

        except Exception as e:
            st.error(f"Error training from document: {e}")

    def _train_from_corrections(self, corrections: List[Dict]):
        """Train the model using user corrections."""
        try:
            # Record feedback in learning database
            from secure_pdf_extractor.learning.database import LearningDatabase
            from secure_pdf_extractor.learning.pattern_learner import PatternLearner

            learning_db = LearningDatabase()
            pattern_learner = PatternLearner()

            for correction in corrections:
                # Record the feedback
                context = st.session_state.get('extracted_text', '')[
                    :500]  # First 500 chars

                learning_db.record_feedback(
                    field_name=correction['field_name'],
                    original_value=correction['original_value'],
                    correct_value=correction['corrected_value'],
                    context=context,
                    pdf_name=correction['pdf_name']
                )

                # Learn new patterns
                pattern_learner.learn_from_feedback(
                    field_type=correction['field_name'],
                    correct_value=correction['corrected_value'],
                    context=context
                )

            st.success(
                f"â‚¬Â¦ Model learned from {len(corrections)} corrections!")

        except Exception as e:
            st.warning(f"Note: Could not record advanced learning data: {e}")

    def _save_training_document(self, pdf_name: str, extracted_text: str, training_data: Dict):
        """Save document data for training purposes."""
        try:
            # Create training data directory
            training_dir = Path("data/user_training")
            training_dir.mkdir(parents=True, exist_ok=True)

            # Create training entry
            training_entry = {
                "pdf_name": pdf_name,
                "timestamp": datetime.now().isoformat(),
                "extracted_text": extracted_text[:2000],  # Limit text size
                "fields": training_data,
                "source": "user_document"
            }

            # Save to JSON file
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_{timestamp}_{pdf_name.replace('.pdf', '')}.json"
            filepath = training_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_entry, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Warning: Could not save training document: {e}")

    def _learn_patterns_from_document(self, training_data: Dict, context: str):
        """Learn extraction patterns from document data."""
        try:
            from secure_pdf_extractor.learning.pattern_learner import PatternLearner

            pattern_learner = PatternLearner()

            for field_name, value in training_data.items():
                if value and value.strip():  # Only learn from non-empty values
                    pattern_learner.learn_from_feedback(
                        field_type=field_name,
                        correct_value=value,
                        context=context[:500]  # Limit context size
                    )

        except Exception as e:
            print(f"Warning: Could not learn patterns: {e}")

    def _fallback_field_extraction(self, text: str) -> Dict[str, str]:
        """Fallback field extraction using basic regex patterns."""
        import re

        fields = {
            'date': '',
            'angebot': '',
            'company_name': '',
            'sender_address': ''
        }

        # Simple date extraction
        date_patterns = [
            r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',
            r'\b\d{4}-\d{1,2}-\d{1,2}\b'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                fields['date'] = match.group()
                break

        # Simple angebot extraction
        angebot_patterns = [
            r'angebot[^\w]*[:\s#-]\s*([A-Z0-9\-/\.#_]{3,20})',
            r'quote[^\w]*[:\s#-]\s*([A-Z0-9\-/\.#_]{3,20})',
            r'([A-Z0-9]{6,20})\s*[-\s]+.*?angebot'
        ]
        for pattern in angebot_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['angebot'] = match.group(1)
                break

        # Simple company extraction (first non-empty line)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            fields['company_name'] = lines[0]

        # Simple address extraction (first few lines)
        if len(lines) > 1:
            fields['sender_address'] = '\n'.join(lines[1:4])

        return fields

    def _simple_fallback_extraction(self, text: str) -> Dict[str, str]:
        """Very simple fallback extraction."""
        import re

        # Even simpler patterns
        date_match = re.search(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', text)
        angebot_match = re.search(r'\b([A-Z0-9]{6,15})\b', text)

        lines = [line.strip() for line in text.split('\n')[:5] if line.strip()]

        return {
            'date': date_match.group() if date_match else '',
            'angebot': angebot_match.group() if angebot_match else '',
            'company_name': lines[0] if lines else '',
            'sender_address': '\n'.join(lines[1:3]) if len(lines) > 1 else ''
        }

    def render_settings(self):
        """Enhanced settings interface with database diagnostics and comprehensive configuration."""
        st.header("âš™ï¸ Settings & Diagnostics")

        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(
            ["ðŸ”§ Extraction Settings", "ðŸ—ƒï¸ Database Diagnostics", "ðŸ“Š Export Settings", "ðŸ§  Learning Settings"])

        # ============= EXTRACTION SETTINGS TAB =============
        with tab1:
            st.subheader("Extraction Configuration")

            col1, col2 = st.columns(2)

            with col1:
                use_basic = st.checkbox(
                    "Use Basic Extraction",
                    key="use_basic_extraction",
                    value=st.session_state.get("use_basic_extraction", True),
                    help="Simple regex-based extraction without adaptive learning"
                )

                auto_language = st.checkbox(
                    "Auto Language Detection",
                    key="auto_language_detection",
                    value=st.session_state.get(
                        "auto_language_detection", True),
                    help="Automatically detect document language for optimal extraction"
                )

                validate_results = st.checkbox(
                    "Validate Results",
                    key="validate_results_checkbox",
                    value=st.session_state.get("validate_results", True),
                    help="Validate extracted field values before display"
                )

            with col2:
                confidence_boost = st.slider(
                    "Confidence Boost",
                    min_value=0.0,
                    max_value=0.5,
                    value=st.session_state.get("confidence_boost", 0.1),
                    step=0.05,
                    key="confidence_boost_slider",
                    help="Boost confidence scores for extracted fields"
                )

                default_language = st.selectbox(
                    "Default Language",
                    options=["de", "en"],
                    index=0,
                    key="default_language",
                    help="Default language when auto-detection fails"
                )

                receiver_company = st.text_input(
                    "Receiver Company",
                    value=st.session_state.get(
                        "receiver_company", ""),
                    key="receiver_company_input",
                    help="Company name to filter out from sender detection"
                )

        # ============= DATABASE DIAGNOSTICS TAB =============
        with tab2:
            st.subheader("ðŸ” Database Diagnostics")

            # Database status overview
            col1, col2 = st.columns(2)

            with col1:
                if st.button("ðŸ” Diagnose Learning Database", key="diagnose_learning_db"):
                    self.diagnose_database_connection()

            with col2:
                if st.button("ðŸ”„ Reinitialize Learning Systems", key="reinit_learning"):
                    try:
                        self._init_learning_systems()
                        st.success("âœ… Learning systems reinitialized")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"âŒ Reinitialization failed: {e}")

            # Database quick stats
            if hasattr(self, 'learning_db') and self.learning_db:
                try:
                    stats = self.learning_db.get_table_correction_stats()
                    pattern_stats = {}
                    if hasattr(self, 'pattern_learner') and self.pattern_learner:
                        pattern_stats = self.pattern_learner.get_pattern_statistics()

                    st.subheader("ðŸ“Š Database Status")

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(
                        4)

                    with metric_col1:
                        st.metric("Table Corrections", stats.get(
                            'total_table_corrections', 0))

                    with metric_col2:
                        st.metric("Documents Processed", stats.get(
                            'documents_with_table_corrections', 0))

                    with metric_col3:
                        st.metric("Learned Patterns",
                                  pattern_stats.get('total_patterns', 0))

                    with metric_col4:
                        confidence_avg = stats.get(
                            'average_confidence_improvement', 0)
                        st.metric("Avg Confidence Boost",
                                  f"{confidence_avg:.1%}")

                except Exception as e:
                    st.warning(f"âš ï¸ Could not load database stats: {e}")

            # Database management
            st.subheader("ðŸ—‚ï¸ Database Management")

            danger_col1, danger_col2 = st.columns(2)

            with danger_col1:
                if st.button("ðŸ—‘ï¸ Clear All Learning Data", key="clear_learning_data"):
                    if st.session_state.get("confirm_clear_data", False):
                        if hasattr(self, 'learning_db') and self.learning_db:
                            try:
                                self.learning_db.clear_all_corrections()
                                st.success("âœ… All learning data cleared")
                                st.session_state["confirm_clear_data"] = False
                            except Exception as e:
                                st.error(f"âŒ Could not clear data: {e}")
                        else:
                            st.error("âŒ Learning database not available")
                    else:
                        st.session_state["confirm_clear_data"] = True
                        st.warning(
                            "âš ï¸ Click again to confirm deletion of ALL learning data")

            with danger_col2:
                if st.button("ðŸ“¥ Backup Database", key="backup_db"):
                    try:
                        import shutil
                        from datetime import datetime

                        if hasattr(self, 'learning_db') and self.learning_db:
                            backup_name = f"learning_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                            shutil.copy2(self.learning_db.db_path, backup_name)
                            st.success(
                                f"âœ… Database backed up as: {backup_name}")
                        else:
                            st.error("âŒ Learning database not available")
                    except Exception as e:
                        st.error(f"âŒ Backup failed: {e}")

        # ============= EXPORT SETTINGS TAB =============
        with tab3:
            st.subheader("ðŸ“¤ Export Configuration")

            col1, col2 = st.columns(2)

            with col1:
                export_format = st.selectbox(
                    "Default Export Format",
                    options=["Excel", "CSV", "JSON"],
                    index=st.session_state.get("export_format_index", 0),
                    key="export_format_selector",
                    help="Select the default export file format"
                )

                include_confidence = st.checkbox(
                    "Include Confidence Scores",
                    key="include_confidence_scores",
                    value=st.session_state.get(
                        "include_confidence_scores", True),
                    help="Include field-level confidence scores in exported files"
                )

                include_timestamps = st.checkbox(
                    "Include Timestamps",
                    key="include_timestamps",
                    value=st.session_state.get("include_timestamps", True),
                    help="Include processing timestamps in exports"
                )

            with col2:
                auto_filename = st.checkbox(
                    "Auto-generate Filenames",
                    key="auto_filename",
                    value=st.session_state.get("auto_filename", True),
                    help="Automatically generate export filenames with timestamps"
                )

                export_original_tables = st.checkbox(
                    "Export Original Tables",
                    key="export_original_tables",
                    value=st.session_state.get(
                        "export_original_tables", False),
                    help="Include original extracted tables in exports"
                )

                compress_exports = st.checkbox(
                    "Compress Large Exports",
                    key="compress_exports",
                    value=st.session_state.get("compress_exports", False),
                    help="Compress exports larger than 10MB"
                )

        # ============= LEARNING SETTINGS TAB =============
        with tab4:
            st.subheader("ðŸ§  Learning Configuration")

            col1, col2 = st.columns(2)

            with col1:
                enable_pattern_learning = st.checkbox(
                    "Enable Pattern Learning",
                    key="enable_pattern_learning",
                    value=st.session_state.get(
                        "enable_pattern_learning", True),
                    help="Learn extraction patterns from user corrections"
                )

                auto_apply_corrections = st.checkbox(
                    "Auto-apply Saved Corrections",
                    key="auto_apply_corrections",
                    value=st.session_state.get("auto_apply_corrections", True),
                    help="Automatically apply previously saved corrections"
                )

                learning_threshold = st.slider(
                    "Learning Confidence Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=st.session_state.get("learning_threshold", 0.7),
                    step=0.05,
                    key="learning_threshold_slider",
                    help="Minimum confidence for learned patterns"
                )

            with col2:
                max_patterns_per_field = st.number_input(
                    "Max Patterns per Field",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get("max_patterns_per_field", 10),
                    key="max_patterns_input",
                    help="Maximum learned patterns to store per field type"
                )

                pattern_update_frequency = st.selectbox(
                    "Pattern Update Frequency",
                    options=["Immediate", "After 5 corrections",
                             "After 10 corrections", "Manual only"],
                    index=st.session_state.get(
                        "pattern_update_frequency_index", 0),
                    key="pattern_update_frequency",
                    help="How often to update learned patterns"
                )

            # Pattern learning statistics
            if hasattr(self, 'pattern_learner') and self.pattern_learner:
                try:
                    stats = self.pattern_learner.get_pattern_statistics()
                    if stats['total_patterns'] > 0:
                        st.subheader("ðŸ“ˆ Learning Statistics")

                        stats_col1, stats_col2, stats_col3 = st.columns(3)

                        with stats_col1:
                            st.metric("Total Patterns",
                                      stats['total_patterns'])

                        with stats_col2:
                            st.metric("Field Types", len(
                                stats['patterns_by_type']))

                        with stats_col3:
                            if stats['top_patterns']:
                                best_success = max(p[4] if len(
                                    p) > 4 else 0 for p in stats['top_patterns'])
                                st.metric("Best Success Rate",
                                          f"{best_success:.1%}")
                            else:
                                st.metric("Best Success Rate", "0%")

                except Exception as e:
                    st.warning(f"âš ï¸ Could not load pattern statistics: {e}")

        # ============= SAVE SETTINGS BUTTON =============
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("ðŸ’¾ Save All Settings", key="save_all_settings", use_container_width=True):
                # Save all settings to session state
                settings_to_save = {
                    "use_basic_extraction": use_basic,
                    "auto_language_detection": auto_language,
                    "confidence_boost": confidence_boost,
                    "validate_results": validate_results,
                    "default_language": default_language,
                    "receiver_company": receiver_company,
                    "export_format": export_format,
                    "export_format_index": ["Excel", "CSV", "JSON"].index(export_format),
                    "include_confidence_scores": include_confidence,
                    "include_timestamps": include_timestamps,
                    "auto_filename": auto_filename,
                    "export_original_tables": export_original_tables,
                    "compress_exports": compress_exports,
                    "enable_pattern_learning": enable_pattern_learning,
                    "auto_apply_corrections": auto_apply_corrections,
                    "learning_threshold": learning_threshold,
                    "max_patterns_per_field": max_patterns_per_field,
                    "pattern_update_frequency": pattern_update_frequency,
                    "pattern_update_frequency_index": ["Immediate", "After 5 corrections", "After 10 corrections", "Manual only"].index(pattern_update_frequency)
                }

                for key, value in settings_to_save.items():
                    st.session_state[key] = value

                st.success("âœ… All settings saved successfully!")

                # Optional: Save to file for persistence
                try:
                    import json
                    settings_file = self.base_path / "user_settings.json"
                    with open(settings_file, 'w') as f:
                        json.dump(settings_to_save, f, indent=2)
                    st.info(f"ðŸ’¾ Settings also saved to {settings_file}")
                except Exception as e:
                    st.warning(f"âš ï¸ Could not save settings to file: {e}")


    def run_test_py_fallback(self, pdf_path: str, mode: str = 'full') -> dict:
        """
        Run test.py as fallback extraction method - COMPLETELY INDEPENDENT from main logic.
        
        Args:
            pdf_path: Path to the PDF file
            mode: 'full' for both fields and tables, 'fields' for fields only, 'tables' for tables only
        
        Returns:
            dict: Results with 'success', 'data', 'error', 'command', 'stdout', 'stderr'
        """
        try:
            # Determine the command based on mode - test.py is in the parent directory
            test_py_path = self.base_path.parent / "test.py"
            
            if mode == 'fields':
                cmd = [sys.executable, str(test_py_path), "--pdf", pdf_path, "--fields-only", "--output", "fallback_output.json"]
            elif mode == 'tables':
                cmd = [sys.executable, str(test_py_path), "--pdf", pdf_path, "--table-only", "--output", "fallback_output.json"]
            elif mode == 'full':
                # Run test.py without any restrictions to get both fields and tables
                cmd = [sys.executable, str(test_py_path), "--pdf", pdf_path, "--output", "fallback_output.json"]
            else:
                return {
                    'success': False,
                    'data': None,
                    'error': f"Invalid mode: {mode}",
                    'command': "",
                    'stdout': "",
                    'stderr': ""
                }
            
            # Store command for logging
            command_str = " ".join(cmd)
            
            # Run the command in the secure_pdf_extractor directory (where test.py expects to be run)
            working_dir = self.base_path.parent  # Go up one level from gui to secure_pdf_extractor
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=str(working_dir)
            )
            
            if result.returncode == 0:
                # Try to read the output JSON file first
                output_file = working_dir / "fallback_output.json"
                json_data = None
                
                if output_file.exists():
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        # Clean up the temporary file
                        output_file.unlink()
                    except Exception as file_error:
                        st.warning(f"Could not read output file: {file_error}")
                
                # If no output file, try to parse JSON from stdout
                if json_data is None:
                    stdout_lines = result.stdout.strip().split('\n')
                    
                    # Look for JSON in the output (check multiple patterns)
                    for line in stdout_lines:
                        line = line.strip()
                        if line.startswith('{') or line.startswith('['):
                            try:
                                json_data = json.loads(line)
                                break
                            except json.JSONDecodeError:
                                continue
                    
                    # Also check for lines that might contain JSON after other text
                    if json_data is None:
                        for line in stdout_lines:
                            # Look for JSON patterns in the line
                            json_start = line.find('{')
                            json_start_list = line.find('[')
                            
                            start_pos = -1
                            if json_start >= 0 and json_start_list >= 0:
                                start_pos = min(json_start, json_start_list)
                            elif json_start >= 0:
                                start_pos = json_start
                            elif json_start_list >= 0:
                                start_pos = json_start_list
                            
                            if start_pos >= 0:
                                try:
                                    json_data = json.loads(line[start_pos:])
                                    break
                                except json.JSONDecodeError:
                                    continue
                
                return {
                    'success': True,
                    'data': json_data,
                    'error': None,
                    'command': command_str,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'data': None,
                    'error': f"test.py exited with code {result.returncode}",
                    'command': command_str,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'data': None,
                'error': "test.py execution timed out (120s)",
                'command': command_str if 'command_str' in locals() else "Unknown command",
                'stdout': "",
                'stderr': "Timeout"
            }
        except Exception as e:
            return {
                'success': False,
                'data': None,
                'error': str(e),
                'command': command_str if 'command_str' in locals() else "Unknown command",
                'stdout': "",
                'stderr': str(e)
            }

    def save_fallback_results(self, filename: str, data: dict, mode: str) -> str:
        """
        Save fallback results to JSON files.
        
        Args:
            filename: Original PDF filename
            data: Extracted data
            mode: 'fields' or 'tables'
        
        Returns:
            str: Path to saved file
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = self.base_path / "extracted_results"
            results_dir.mkdir(exist_ok=True)
            
            # Generate filename
            base_name = Path(filename).stem
            if mode == 'fields':
                output_file = results_dir / f"{base_name}_fields.json"
            else:
                output_file = results_dir / f"{base_name}_tables.json"
            
            # Save data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return str(output_file)
            
        except Exception as e:
            st.error(f"Failed to save results: {e}")
            return ""

    def add_run_log_entry(self, command: str, success: bool, stdout: str, stderr: str):
        """Add entry to run log in session state."""
        if 'run_log' not in st.session_state:
            st.session_state.run_log = []
        
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'command': command,
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        }
        
        # Keep only last 50 entries
        st.session_state.run_log.append(entry)
        if len(st.session_state.run_log) > 50:
            st.session_state.run_log = st.session_state.run_log[-50:]

    def render_run_log_panel(self):
        """Render collapsible run log panel for debugging."""
        if 'run_log' not in st.session_state or not st.session_state.run_log:
            return
            
        with st.expander("🔍 Run Log (Debugging)", expanded=False):
            st.markdown("### Recent test.py Commands")
            
            for i, entry in enumerate(reversed(st.session_state.run_log[-10:])):  # Show last 10
                status_icon = "✅" if entry['success'] else "❌"
                
                with st.container():
                    st.markdown(f"**{status_icon} {entry['timestamp']}**")
                    st.code(entry['command'], language='bash')
                    
                    if entry['stdout']:
                        with st.expander("📤 Standard Output", expanded=False):
                            st.text(entry['stdout'])
                    
                    if entry['stderr']:
                        with st.expander("⚠️ Standard Error", expanded=False):
                            st.text(entry['stderr'])
                    
                    st.markdown("---")

    def display_tables_beautifully(self, tables: list):
        """
        Display tables in a beautiful, readable format using Streamlit.
        
        Args:
            tables: List of table data (various formats supported)
        """
        if not tables:
            st.info("📊 No tables found in this document")
            return
        
        st.success(f"✅ Found {len(tables)} table(s)")
        
        try:
            # Handle different table formats
            for i, table in enumerate(tables, 1):
                with st.expander(f"📋 Table {i} - {table.get('title', 'Extracted Table')}", expanded=True):
                    
                    if isinstance(table, dict):
                        # AI-extracted format with headers and rows
                        if 'headers' in table and 'rows' in table:
                            headers = table.get('headers', [])
                            rows = table.get('rows', [])
                            
                            if headers and rows:
                                # Create a proper DataFrame
                                
                                # Prepare data for DataFrame
                                table_data = []
                                for row_idx, row in enumerate(rows):
                                    row_dict = {}
                                    for col_idx, cell in enumerate(row):
                                        if col_idx < len(headers):
                                            col_name = str(headers[col_idx]).strip()
                                        else:
                                            col_name = f"Column_{col_idx+1}"
                                        
                                        # Clean and format cell data
                                        cell_value = str(cell).strip() if cell is not None else ""
                                        row_dict[col_name] = cell_value
                                    table_data.append(row_dict)
                                
                                if table_data:
                                    df = pd.DataFrame(table_data)
                                    
                                    # Display with nice formatting
                                    st.markdown("**📊 Table Data:**")
                                    st.dataframe(
                                        df, 
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            col: st.column_config.TextColumn(
                                                width="medium",
                                                help=f"Data from column: {col}"
                                            ) for col in df.columns
                                        }
                                    )
                                    
                                    # Show table statistics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Rows", len(rows))
                                    with col2:
                                        st.metric("Columns", len(headers))
                                    with col3:
                                        filled_cells = sum(1 for row in rows for cell in row if cell and str(cell).strip())
                                        total_cells = len(rows) * len(headers)
                                        completion = (filled_cells / total_cells * 100) if total_cells > 0 else 0
                                        st.metric("Data Completion", f"{completion:.1f}%")
                                        
                                    # Show raw data option
                                    if st.checkbox(f"Show raw data for Table {i}", key=f"raw_data_{i}"):
                                        st.json({"headers": headers, "rows": rows[:3]})  # First 3 rows only
                                else:
                                    st.warning("No valid data rows found")
                            else:
                                st.warning("Table has no headers or rows")
                        
                        # PDF-extracted format with page info
                        elif 'page' in table:
                            st.markdown(f"**📄 From Page {table.get('page', 1)}**")
                            
                            # Display as formatted table
                            row_data = {k: [v] for k, v in table.items() if k not in ['page', 'table']}
                            if row_data:
                                df = pd.DataFrame(row_data)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                st.warning("No data found in this table")
                        
                        # Generic dictionary format
                        else:
                            st.markdown("**🔍 Table Structure:**")
                            # Try to create a nice display
                            if len(table) > 0:
                                # Convert to DataFrame if possible
                                try:
                                    df = pd.DataFrame([table])
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                                except:
                                    # Fallback to JSON
                                    st.json(table)
                            else:
                                st.warning("Empty table")
                    
                    elif isinstance(table, list) and table:
                        # List format - treat as rows
                        st.markdown("**📊 Table Data:**")
                        
                        if isinstance(table[0], dict):
                            # List of dictionaries
                            df = pd.DataFrame(table)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            # Show statistics
                            st.metric("Records", len(table))
                            
                        elif isinstance(table[0], list):
                            # List of lists (rows and columns) - like your data format
                            st.markdown("**📋 Structured Table:**")
                            
                            # First row is usually headers
                            if len(table) > 0:
                                headers = [str(cell) for cell in table[0]]
                                data_rows = table[1:] if len(table) > 1 else []
                                
                                if data_rows:
                                    # Create DataFrame from rows
                                    table_data = []
                                    for row in data_rows:
                                        row_dict = {}
                                        for col_idx, cell in enumerate(row):
                                            if col_idx < len(headers):
                                                col_name = headers[col_idx]
                                            else:
                                                col_name = f"Column_{col_idx+1}"
                                            row_dict[col_name] = str(cell) if cell else ""
                                        table_data.append(row_dict)
                                    
                                    df = pd.DataFrame(table_data)
                                    st.dataframe(
                                        df, 
                                        use_container_width=True, 
                                        hide_index=True,
                                        column_config={
                                            col: st.column_config.TextColumn(
                                                width="medium"
                                            ) for col in df.columns
                                        }
                                    )
                                    
                                    # Show table statistics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Data Rows", len(data_rows))
                                    with col2:
                                        st.metric("Columns", len(headers))
                                    with col3:
                                        filled_cells = sum(1 for row in data_rows for cell in row if cell and str(cell).strip())
                                        total_cells = len(data_rows) * len(headers)
                                        completion = (filled_cells / total_cells * 100) if total_cells > 0 else 0
                                        st.metric("Data Completion", f"{completion:.1f}%")
                                else:
                                    # Only headers, no data
                                    st.info("Table contains only headers, no data rows")
                                    st.write("**Headers:**")
                                    for idx, header in enumerate(headers):
                                        st.write(f"{idx + 1}. {header}")
                            else:
                                st.warning("Empty table")
                        else:
                            # Simple list of values
                            for idx, item in enumerate(table[:10]):  # Show first 10 items
                                st.write(f"{idx + 1}. {item}")
                            if len(table) > 10:
                                st.info(f"... and {len(table) - 10} more items")
                    
                    else:
                        # Unknown format
                        st.markdown("**⚠️ Unknown Table Format**")
                        st.write(f"Type: {type(table)}")
                        st.write(table)
                    
                    # Add separator between tables
                    if i < len(tables):
                        st.markdown("---")
                        
        except Exception as e:
            st.error(f"❌ Error displaying tables: {e}")
            st.error(f"Error details: {str(e)}")
            
            # Enhanced fallback display
            st.markdown("**🔧 Raw Table Data (Fallback Display):**")
            for i, table in enumerate(tables, 1):
                with st.expander(f"Raw Table {i}", expanded=False):
                    st.json(table)

    def render_parameter_controls(self, extracted_fields: dict, document_hash: str):
        """
        Render per-parameter controls for editing and adding fields.
        
        Args:
            extracted_fields: Current extracted fields
            document_hash: Document hash for saving corrections
        """
        # Only render if we have fields to work with
        if not extracted_fields:
            return
            
        st.subheader("📝 Field Controls")
        
        # Generate unique timestamp for this render to avoid key conflicts
        import time
        unique_suffix = str(int(time.time() * 1000))[-8:]  # Last 8 digits of timestamp
        
        # Add Parameter section
        st.markdown("#### Add New Parameter")
        col1, col2, col3 = st.columns([3, 3, 2])
        
        with col1:
            new_param_name = st.text_input(
                "Parameter Name", 
                key=f"new_param_name_{document_hash[:8]}_{unique_suffix}",
                placeholder="e.g., InvoiceNumber"
            )
        
        with col2:
            new_param_value = st.text_input(
                "Parameter Value", 
                key=f"new_param_value_{document_hash[:8]}_{unique_suffix}",
                placeholder="e.g., INV-2024-001"
            )
        
        with col3:
            if st.button("➕ Add Parameter", key=f"add_param_{document_hash[:8]}_{unique_suffix}"):
                if new_param_name and new_param_value:
                    # Add to extracted fields
                    extracted_fields[new_param_name] = new_param_value
                    
                    # Save correction to database if available
                    try:
                        if hasattr(self, 'learning_db') and self.learning_db:
                            self.learning_db.save_correction(
                                document_hash=document_hash,
                                pdf_name="current_document.pdf",
                                field_name=new_param_name,
                                corrected_value=new_param_value
                            )
                        st.success(f"✅ Added parameter: {new_param_name}")
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Parameter added but not saved to database: {e}")
                else:
                    st.warning("Please provide both name and value")
        
        # Display existing parameters (read-only)
        if extracted_fields:
            st.markdown("#### Current Extracted Parameters")
            
            for field_name, current_value in extracted_fields.items():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"**{field_name}:**")
                
                with col2:
                    st.text(str(current_value) if current_value else "(empty)")
            
            st.info("💡 Use the 'Add Parameter' section above to add new fields or override existing ones.")


    def export_comprehensive_excel(self, fields: Dict[str, str] = None, tables: list = None, filename_prefix: str = "extracted_data"):
        """Export extracted fields and tables to comprehensive Excel with multiple sheets."""
        try:
            import io
            from datetime import datetime
            
            # Create Excel file in memory
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                
                # Export Fields Sheet
                if fields:
                    # Create dynamic field data
                    field_data = []
                    confidence_scores = getattr(st.session_state, 'confidence_scores', {})
                    
                    for field_name, field_value in fields.items():
                        confidence = confidence_scores.get(field_name, 0.0)
                        field_data.append({
                            'Field': field_name,
                            'Value': str(field_value) if field_value else '',
                            'Confidence': f"{confidence:.1%}",
                            'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
                    if field_data:
                        fields_df = pd.DataFrame(field_data)
                        fields_df.to_excel(writer, index=False, sheet_name='Extracted_Fields')
                
                # Export Tables Sheets
                if tables:
                    for i, table in enumerate(tables, 1):
                        sheet_name = f'Table_{i}'
                        
                        try:
                            if isinstance(table, dict):
                                # Handle AI-extracted format with headers and rows
                                if 'headers' in table and 'rows' in table:
                                    headers = table.get('headers', [])
                                    rows = table.get('rows', [])
                                    
                                    if headers and rows:
                                        # Convert to DataFrame
                                        table_data = []
                                        for row in rows:
                                            if len(row) <= len(headers):
                                                row_dict = {headers[j]: (str(cell) if cell else "") for j, cell in enumerate(row)}
                                            else:
                                                row_dict = {}
                                                for j, cell in enumerate(row):
                                                    col_name = headers[j] if j < len(headers) else f"Column_{j+1}"
                                                    row_dict[col_name] = str(cell) if cell else ""
                                            table_data.append(row_dict)
                                        
                                        table_df = pd.DataFrame(table_data)
                                        table_df.to_excel(writer, index=False, sheet_name=sheet_name)
                                
                                # Handle PDF-extracted format with page info
                                elif 'page' in table:
                                    row_data = {k: [v] for k, v in table.items() if k not in ['page', 'table']}
                                    if row_data:
                                        table_df = pd.DataFrame(row_data)
                                        table_df.to_excel(writer, index=False, sheet_name=sheet_name)
                                
                                # Handle generic dictionary format
                                else:
                                    table_df = pd.DataFrame([table])
                                    table_df.to_excel(writer, index=False, sheet_name=sheet_name)
                            
                            elif isinstance(table, list) and table:
                                # Handle list format
                                if isinstance(table[0], dict):
                                    table_df = pd.DataFrame(table)
                                    table_df.to_excel(writer, index=False, sheet_name=sheet_name)
                                else:
                                    # Handle list of lists
                                    table_df = pd.DataFrame(table)
                                    table_df.to_excel(writer, index=False, sheet_name=sheet_name)
                        
                        except Exception as table_error:
                            # If table export fails, create an error sheet
                            error_df = pd.DataFrame([{
                                'Error': f'Failed to export table {i}',
                                'Details': str(table_error),
                                'Raw_Data': str(table)[:500]  # First 500 chars
                            }])
                            error_df.to_excel(writer, index=False, sheet_name=f'Table_{i}_Error')
                
                # Add Summary Sheet
                summary_data = {
                    'Export_Info': [
                        'Export Date',
                        'Fields Count',
                        'Tables Count',
                        'Software Version'
                    ],
                    'Value': [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(fields) if fields else 0,
                        len(tables) if tables else 0,
                        'PDF Extractor v1.0'
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, index=False, sheet_name='Export_Summary')
            
            output.seek(0)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename_prefix}_{timestamp}.xlsx"
            
            # Provide download button
            st.download_button(
                label="📊 Download Comprehensive Excel File",
                data=output.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Also save to results folder
            try:
                results_dir = self.base_path / "extracted_results"
                results_dir.mkdir(exist_ok=True)
                excel_path = results_dir / filename
                
                with open(excel_path, 'wb') as f:
                    f.write(output.getvalue())
                
                st.success(f"✅ Excel file ready for download and saved to: {excel_path.name}")
                
            except Exception as save_error:
                st.success("✅ Excel file ready for download!")
                st.info(f"ℹ️ Could not save to results folder: {save_error}")

        except Exception as e:
            st.error(f"❌ Failed to export Excel file: {e}")
            st.error(f"Error details: {str(e)}")
            
            # Provide fallback - save as CSV
            try:
                if fields:
                    csv_data = pd.DataFrame([fields])
                    csv_filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    st.download_button(
                        label="📄 Download as CSV (Fallback)",
                        data=csv_data.to_csv(index=False),
                        file_name=csv_filename,
                        mime="text/csv"
                    )
                    st.info("Excel export failed, but CSV download is available.")
            except Exception:
                st.error("Both Excel and CSV export failed.")


def main():
    """Main function to run the Streamlit app."""
    app = PDFExtractorApp()
    app.run()


if __name__ == "__main__":
    main()



# help me to fine tune this model...i have 50 pdfs with answers in excel sheet...so 50 pdfs withy 50 excel sheets (answer)....help me to train the model,give me idea ,how to train this mode