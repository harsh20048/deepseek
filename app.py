import streamlit as st
import pdfplumber
import json
from tabulate import tabulate
import os
from datetime import datetime
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile

# Set page config
st.set_page_config(
    page_title="DeepSeek PDF Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional UI
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: none;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .field-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #1f77b4;
    }
    
    .field-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    
    .field-value {
        font-size: 1.1rem;
        color: #333;
        background: rgba(255,255,255,0.8);
        padding: 0.8rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        word-break: break-word;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Progress styling */
    .progress-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 30px;
        border-radius: 15px;
        overflow: hidden;
        margin: 1rem 0;
        position: relative;
    }
    
    .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Upload area styling */
    .uploadedFile {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

def load_model():
    """Load the local DeepSeek model"""
    try:
        with st.spinner("Loading DeepSeek model..."):
            model_path = "C:/Users/HARSH/OneDrive/Desktop/pdf/model"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )
            
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF with enhanced error handling and structure preservation"""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            total_pages = len(pdf.pages)
            
            if total_pages == 0:
                st.error("⚠️ PDF appears to be empty or corrupted")
                return None
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            pages_with_text = 0
            for i, page in enumerate(pdf.pages, 1):
                # Try different extraction methods for better structure
                page_text = None
                
                # Method 1: Try to preserve layout with extract_text()
                try:
                    page_text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3)
                except:
                    pass
                
                # Method 2: Fallback to standard extraction
                if not page_text:
                    try:
                        page_text = page.extract_text()
                    except:
                        pass
                
                # Method 3: Try character-level extraction for complex layouts
                if not page_text:
                    try:
                        chars = page.chars
                        if chars:
                            # Group characters by lines based on y-coordinate
                            lines = {}
                            for char in chars:
                                y = round(char['y0'], 1)
                                if y not in lines:
                                    lines[y] = []
                                lines[y].append(char)
                            
                            # Sort lines by y-coordinate (top to bottom)
                            sorted_lines = []
                            for y in sorted(lines.keys(), reverse=True):
                                line_chars = sorted(lines[y], key=lambda x: x['x0'])
                                line_text = ''.join([char['text'] for char in line_chars])
                                if line_text.strip():
                                    sorted_lines.append(line_text.strip())
                            
                            page_text = '\n'.join(sorted_lines)
                    except:
                        page_text = ""
                
                if page_text and page_text.strip():
                    # Clean up the text while preserving structure
                    cleaned_text = clean_extracted_text(page_text)
                    text += f"\n--- Page {i} ---\n" + cleaned_text + "\n"
                    pages_with_text += 1
                
                progress_bar.progress(i / total_pages)
                status_text.text(f"Processing page {i}/{total_pages}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Check if we got meaningful text
            if not text.strip():
                st.error("⚠️ **No text could be extracted from this PDF!**")
                st.warning("""
                **Possible reasons:**
                - PDF is scanned/image-based (needs OCR)
                - PDF is password-protected
                - PDF contains only images/graphics
                - PDF is corrupted
                
                **Solutions:**
                - Use OCR tools like Tesseract to convert to text-based PDF
                - Check if you can select text manually in the PDF
                - Try a different PDF viewer to verify content
                """)
                return None
            
            if pages_with_text < total_pages:
                st.warning(f"⚠️ Only {pages_with_text} out of {total_pages} pages contained extractable text")
            
            # Check for potential OCR needs
            if len(text) < 100:  # Very short text might indicate OCR needed
                st.warning("⚠️ Very little text extracted. This might be a scanned document that needs OCR processing.")
            
            return text
            
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        st.info("💡 **Tips for better results:**\n- Ensure PDF is not password-protected\n- Try with a different PDF file\n- Check if PDF contains selectable text")
        return None

def clean_extracted_text(text):
    """Clean and structure extracted PDF text for better readability"""
    import re
    
    if not text:
        return ""
    
    # Split into lines and clean each line
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove excessive whitespace but preserve single spaces
        cleaned_line = re.sub(r'\s+', ' ', line.strip())
        
        # Skip empty lines
        if not cleaned_line:
            continue
            
        # Add line breaks after common German business document patterns
        if re.match(r'^(Sehr geehrte|Vielen Dank|Mit freundlichen|Datum:|Angebot|Rechnung)', cleaned_line, re.IGNORECASE):
            cleaned_lines.append('\n' + cleaned_line)
        elif re.match(r'^(Pos|Position)\s+', cleaned_line, re.IGNORECASE):
            cleaned_lines.append('\n' + cleaned_line)
        elif re.match(r'^\d+\s+', cleaned_line):  # Lines starting with numbers (positions)
            cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append(cleaned_line)
    
    # Join lines and add proper spacing
    result = '\n'.join(cleaned_lines)
    
    # Add spacing around key sections
    result = re.sub(r'(Angebot Nr\.?\s*\d+)', r'\n\1\n', result, flags=re.IGNORECASE)
    result = re.sub(r'(Datum:\s*\d+\.\d+\.\d+)', r'\1\n', result, flags=re.IGNORECASE)
    result = re.sub(r'(Sehr geehrte.*)', r'\n\1\n', result, flags=re.IGNORECASE)
    
    # Clean up multiple newlines
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result.strip()

def query_local_model(prompt, max_new_tokens=512):
    """Query the local DeepSeek model"""
    try:
        inputs = st.session_state.tokenizer.encode(prompt, return_tensors="pt")
        
        # Check if input is too long and truncate if necessary
        max_input_length = 2048  # Set a reasonable input limit
        if inputs.shape[1] > max_input_length:
            inputs = inputs[:, -max_input_length:]  # Take the last max_input_length tokens
        
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                inputs, 
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
                eos_token_id=st.session_state.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    except Exception as e:
        st.error(f"Error querying model: {e}")
        return "Error processing query"

# Removed regex fallback extractor - using AI only as requested

def truncate_table_json(table_json, max_len=40):
    """Clean and truncate table data for better display"""
    try:
        # Clean the response first
        cleaned_response = table_json.strip()
        
        # Try to find JSON array in the response
        if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
            data = json.loads(cleaned_response)
        else:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', cleaned_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # If no JSON found, try to create a simple table from the text
                lines = cleaned_response.split('\n')
                data = []
                for line in lines:
                    if line.strip() and '|' in line:
                        parts = [part.strip() for part in line.split('|') if part.strip()]
                        if len(parts) >= 2:
                            row = {}
                            for i, part in enumerate(parts):
                                row[f"Column_{i+1}"] = part[:max_len] + "..." if len(part) > max_len else part
                            data.append(row)
        
        # Truncate long values
        for row in data:
            for key, value in row.items():
                if isinstance(value, str) and len(value) > max_len:
                    row[key] = value[:max_len] + "..."
        
        return data
    except Exception as e:
        st.warning(f"Could not parse table data as JSON: {e}")
        # Return a simple fallback
        return [{"Error": "Could not parse table data", "Raw_Response": table_json[:100] + "..."}]

# Main UI with enhanced styling
st.markdown('<h1 class="main-header">🔍 DeepSeek PDF Processor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔒 Process German PDFs with your local DeepSeek model - 100% Offline & Private</p>', unsafe_allow_html=True)

# Add a beautiful info banner
st.markdown("""
<div class="info-card">
    <h3 style="margin-top: 0;">🚀 Pure AI Document Processing</h3>
    <p style="margin-bottom: 0;">
        Extract structured data from German quotations and invoices using only the powerful 
        DeepSeek AI model - no regex fallbacks, pure machine intelligence! All processing 
        happens locally on your machine - your documents never leave your computer!
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">🤖 Model Control Center</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="warning-card">
            <h4 style="margin-top: 0;">⚠️ Model Not Loaded</h4>
            <p style="margin-bottom: 0;">Click below to initialize your local DeepSeek model</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Load DeepSeek Model", type="primary"):
            if load_model():
                st.markdown("""
                <div class="success-card">
                    <h4 style="margin-top: 0;">✅ Success!</h4>
                    <p style="margin-bottom: 0;">Model loaded and ready to process PDFs</p>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()
            else:
                st.markdown("""
                <div class="warning-card">
                    <h4 style="margin-top: 0;">❌ Loading Failed</h4>
                    <p style="margin-bottom: 0;">Please check your model files and try again</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-card">
            <h4 style="margin-top: 0;">✅ Model Ready</h4>
            <p>🔒 Running completely offline</p>
            <p style="margin-bottom: 0;">🚀 Ready to process your PDFs!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.header("📋 Instructions")
    st.markdown("""
    1. **Load Model**: Click the button above to load your local DeepSeek model
    2. **Upload PDF**: Upload a German PDF file using the file uploader
    3. **Process**: Click "Process PDF" to extract structured data
    4. **View Results**: See extracted fields and table data below
    """)
    
    st.header("⚠️ Important Limitations")
    with st.expander("Click to read before processing", expanded=False):
        st.markdown("""
        **Document Requirements:**
        - ✅ **Text-based PDFs only** (you can select text)
        - ❌ **Scanned/image PDFs will fail** (need OCR first)
        - ✅ **German language optimized**
        - ❌ **Other languages may have poor results**
        
        **What Works Best:**
        - Standard German quotations/invoices
        - Simple table layouts
        - Clear, readable text
        - Standard business formats (GmbH, AG, etc.)
        
        **Known Issues:**
        - Complex table layouts may fail
        - AI model may return non-JSON responses for very complex documents
        - Unconventional company naming
        - Multi-column layouts
        - Password-protected PDFs
        - No fallback processing - relies purely on AI model accuracy
        
        **Before Processing:**
        - Verify you can select text in your PDF
        - Check document is in German
        - Ensure standard business format
        """)
    
# Demo section removed - using AI-only approach

# Main content area
if not st.session_state.model_loaded:
    st.warning("⚠️ Please load the DeepSeek model first using the sidebar.")
    st.info("💡 The model will run completely offline using your local files.")
else:
    # File upload section
    st.header("📄 Upload German PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more German PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple German quotation or document PDFs to extract structured data"
    )
    
    if uploaded_files is not None and len(uploaded_files) > 0:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        # Display uploaded files
        for i, file in enumerate(uploaded_files):
            st.write(f"📄 **File {i+1}:** {file.name} ({file.size:,} bytes)")
        
        # Process button
        if st.button("🚀 Process All PDFs", type="primary"):
            # Initialize results storage
            all_results = []
            all_tables = []
            
            # Process each file
            for file_idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"---")
                st.header(f"📄 Processing: {uploaded_file.name}")
                
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Extract text
                    st.info("📖 Extracting text from PDF...")
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    
                    if pdf_text is None:
                        st.error(f"❌ Failed to extract text from {uploaded_file.name}")
                        continue
                    else:
                        st.success(f"✅ Successfully extracted {len(pdf_text)} characters")
                        
                        # Show structured preview of extracted text
                        with st.expander(f"📄 Preview extracted text from {uploaded_file.name}", expanded=False):
                            # Create a nicely formatted preview
                            preview_text = pdf_text[:1500] + "..." if len(pdf_text) > 1500 else pdf_text
                            
                            st.markdown("### 📋 Structured Text Preview")
                            st.markdown("""
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; 
                                       border-left: 4px solid #1f77b4; font-family: 'Courier New', monospace;
                                       white-space: pre-wrap; max-height: 400px; overflow-y: auto;">
                            """ + preview_text.replace('\n', '<br>') + """
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show text statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Characters", f"{len(pdf_text):,}")
                            with col2:
                                lines = len([l for l in pdf_text.split('\n') if l.strip()])
                                st.metric("Text Lines", lines)
                            with col3:
                                words = len(pdf_text.split())
                                st.metric("Word Count", f"{words:,}")
                            with col4:
                                # Estimate reading time (average 200 words per minute)
                                reading_time = max(1, words // 200)
                                st.metric("Reading Time", f"{reading_time} min")
                        
                        # Enhanced field extraction query for unstructured text
                        field_query = """
                        Extract the following fields from this German business document text. 
                        The text may be unstructured with information scattered across lines.
                        
                        Fields to extract:
                        - Date: Look for "Datum:" followed by a date in DD.MM.YYYY format
                        - Angebot: Look for "Angebot Nr." or "Angebot Nr" followed by numbers/letters
                        - SenderCompany: Look for company name with GmbH, AG, UG, etc. (usually at the beginning)
                        - SenderAddress: Look for street address, postal code, and city (often after company name)
                        
                        IMPORTANT RULES:
                        1. Return ONLY valid JSON with these exact keys: Date, Angebot, SenderCompany, SenderAddress
                        2. If a field is not found, use empty string ""
                        3. For SenderAddress, combine street, postal code, and city into one field
                        4. Look carefully through the unstructured text - information may be on separate lines
                        5. Do not include any text outside the JSON object
                        
                        Example format:
                        {
                          "Date": "11.07.2024",
                          "Angebot": "36353",
                          "SenderCompany": "Schlenotronic GmbH",
                          "SenderAddress": "Adam-Opel-Str. 1, 67227 Frankenthal"
                        }
                        """
                        
                        with st.spinner("🤖 Extracting fields with DeepSeek..."):
                            # Truncate PDF text if too long
                            max_pdf_length = 1500  # Limit PDF text length
                            truncated_pdf = pdf_text[:max_pdf_length] + "..." if len(pdf_text) > max_pdf_length else pdf_text
                            
                            full_prompt = f"PDF Content:\n{truncated_pdf}\n\n{field_query}"
                            fields_response = query_local_model(full_prompt, max_new_tokens=256)
                        
                        # Try to parse as JSON, fallback to raw text
                        try:
                            parsed_fields = json.loads(fields_response)
                            # Validate that we got reasonable fields
                            if not isinstance(parsed_fields, dict):
                                raise ValueError("Response is not a dictionary")
                        except Exception as e:
                            st.warning(f"⚠️ AI model returned non-JSON response for {uploaded_file.name}")
                            st.info("💡 Showing raw AI response - no fallback processing")
                            # Only show raw AI response, no fallback processing
                            parsed_fields = {"Raw_Response": fields_response[:500] + "..." if len(fields_response) > 500 else fields_response}
                        
                        # Store results
                        file_result = {
                            "filename": uploaded_file.name,
                            "fields": parsed_fields,
                            "raw_response": fields_response,
                            "text_length": len(pdf_text)
                        }
                        all_results.append(file_result)
                        
                        # Display fields in beautiful cards
                        with st.expander(f"📋 Extracted Data from {uploaded_file.name}", expanded=True):
                            if isinstance(parsed_fields, dict) and "Raw_Response" not in parsed_fields:
                                st.markdown("### 🎯 Document Information")
                                
                                # Create beautiful field cards
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="field-card">
                                        <div class="field-title">📅 Document Date</div>
                                        <div class="field-value">{parsed_fields.get('Date', 'Not found')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown(f"""
                                    <div class="field-card">
                                        <div class="field-title">🏢 Company Name</div>
                                        <div class="field-value">{parsed_fields.get('SenderCompany', 'Not found')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="field-card">
                                        <div class="field-title">📄 Quotation Number</div>
                                        <div class="field-value">{parsed_fields.get('Angebot', 'Not found')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown(f"""
                                    <div class="field-card">
                                        <div class="field-title">📍 Company Address</div>
                                        <div class="field-value">{parsed_fields.get('SenderAddress', 'Not found')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Add extraction quality indicator
                                fields_found = sum(1 for v in parsed_fields.values() if v and str(v).strip() and str(v).strip() != 'Not found')
                                quality_percentage = (fields_found / 4) * 100
                                
                                if quality_percentage >= 75:
                                    quality_color = "#4facfe"
                                    quality_text = "Excellent"
                                elif quality_percentage >= 50:
                                    quality_color = "#667eea"
                                    quality_text = "Good"
                                else:
                                    quality_color = "#fa709a"
                                    quality_text = "Partial"
                                
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {quality_color} 0%, {quality_color}88 100%); 
                                           color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
                                    <h4 style="margin: 0;">🎯 Extraction Quality: {quality_text}</h4>
                                    <p style="margin: 0.5rem 0;">Found {fields_found} out of 4 fields ({quality_percentage:.0f}%)</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            else:
                                # Enhanced fallback display
                                st.markdown("""
                                <div class="warning-card">
                                    <h4 style="margin-top: 0;">⚠️ Raw AI Response</h4>
                                    <p>The AI model returned unstructured data. This might happen with complex documents.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                raw_response = parsed_fields.get("Raw_Response", str(parsed_fields))
                                st.code(raw_response[:500] + "..." if len(raw_response) > 500 else raw_response)
                        
                        # Table extraction query
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
                        
                        with st.spinner("🤖 Extracting table data with DeepSeek..."):
                            # Truncate PDF text if too long
                            max_pdf_length = 1500  # Limit PDF text length
                            truncated_pdf = pdf_text[:max_pdf_length] + "..." if len(pdf_text) > max_pdf_length else pdf_text
                            
                            full_table_prompt = f"PDF Content:\n{truncated_pdf}\n\n{table_query}"
                            table_response = query_local_model(full_table_prompt, max_new_tokens=512)
                        
                        # Process and display table
                        cleaned_table = truncate_table_json(table_response)
                        
                        if cleaned_table:
                            st.success(f"Found {len(cleaned_table)} table rows in {uploaded_file.name}")
                            
                            # Convert to DataFrame for better display
                            import pandas as pd
                            df = pd.DataFrame(cleaned_table)
                            df['Source_File'] = uploaded_file.name  # Add source file column
                            
                            with st.expander(f"📊 Table Data from {uploaded_file.name}", expanded=True):
                                st.markdown("**Extracted Product/Item Information:**")
                                
                                # Display table with better formatting
                                if not df.empty:
                                    # Style the dataframe
                                    styled_df = df.style.set_properties(**{
                                        'background-color': '#f8f9fa',
                                        'color': '#333',
                                        'border': '1px solid #dee2e6'
                                    }).set_table_styles([
                                        {'selector': 'th', 'props': [
                                            ('background-color', '#1f77b4'),
                                            ('color', 'white'),
                                            ('font-weight', 'bold'),
                                            ('text-align', 'center')
                                        ]}
                                    ])
                                    
                                    st.dataframe(df, width='stretch')
                                    
                                    # Show summary statistics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Items", len(df))
                                    with col2:
                                        if 'Menge' in df.columns:
                                            try:
                                                total_qty = sum([float(str(x).replace(',', '.')) for x in df['Menge'] if str(x).replace(',', '.').replace('.', '').isdigit()])
                                                st.metric("Total Quantity", f"{total_qty:.0f}")
                                            except:
                                                st.metric("Total Quantity", "N/A")
                                        else:
                                            st.metric("Total Quantity", "N/A")
                                    with col3:
                                        st.metric("Data Source", uploaded_file.name)
                                else:
                                    st.info("No table data found in this document")
                            
                            # Store table data
                            all_tables.append(df)
                        else:
                            st.warning(f"No table data found in {uploaded_file.name}")
            
            # Enhanced Summary section
            st.markdown("---")
            st.markdown("## 📊 Batch Processing Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{len(all_results)}</div>
                    <div class="metric-label">📄 Files Processed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_tables = sum(len(table) for table in all_tables)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{total_tables}</div>
                    <div class="metric-label">📊 Table Rows</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_chars = sum(result['text_length'] for result in all_results)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{total_chars:,}</div>
                    <div class="metric-label">📝 Characters</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                # Calculate average extraction quality
                total_fields = 0
                found_fields = 0
                for result in all_results:
                    if isinstance(result['fields'], dict) and "Raw_Response" not in result['fields']:
                        total_fields += 4
                        found_fields += sum(1 for v in result['fields'].values() if v and str(v).strip() and str(v).strip() != 'Not found')
                
                avg_quality = (found_fields / total_fields * 100) if total_fields > 0 else 0
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{avg_quality:.0f}%</div>
                    <div class="metric-label">🎯 Avg Quality</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Combined results
            if all_results:
                st.subheader("📋 All Extracted Fields Summary")
                
                # Create a summary table of all fields
                summary_data = []
                for result in all_results:
                    fields = result['fields']
                    if isinstance(fields, dict) and "Raw_Response" not in fields:
                        summary_data.append({
                            "File": result['filename'],
                            "Date": fields.get('Date', 'Not found'),
                            "Quotation #": fields.get('Angebot', 'Not found'),
                            "Company": fields.get('SenderCompany', 'Not found'),
                            "Address": fields.get('SenderAddress', 'Not found')[:50] + "..." if len(str(fields.get('SenderAddress', ''))) > 50 else fields.get('SenderAddress', 'Not found')
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, width='stretch')
                else:
                    st.info("No structured field data could be extracted from the documents")
            
            # Combined tables
            if all_tables:
                st.subheader("📊 All Table Data Combined")
                combined_df = pd.concat(all_tables, ignore_index=True)
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", len(combined_df))
                with col2:
                    st.metric("Total Files", len(all_tables))
                with col3:
                    st.metric("Columns", len(combined_df.columns))
                with col4:
                    unique_sources = combined_df['Source_File'].nunique() if 'Source_File' in combined_df.columns else 0
                    st.metric("Unique Sources", unique_sources)
                
                # Display the combined table with better formatting
                st.markdown("**Combined Product/Item Data from All Documents:**")
                st.dataframe(combined_df, width='stretch')
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Tables as CSV",
                        data=csv,
                        file_name="all_extracted_tables.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Create summary JSON
                    summary_data = {
                        "files_processed": len(all_results),
                        "total_table_rows": len(combined_df),
                        "files": all_results
                    }
                    json_data = json.dumps(summary_data, indent=2)
                    st.download_button(
                        label="📥 Download Summary as JSON",
                        data=json_data,
                        file_name="processing_summary.json",
                        mime="application/json"
                    )
            
            # Beautiful completion message
            st.markdown("""
            <div class="success-card" style="text-align: center; margin: 2rem 0;">
                <h2 style="margin-top: 0;">🎉 Processing Complete!</h2>
                <p style="font-size: 1.2rem;">All documents have been successfully processed</p>
                <p style="margin-bottom: 0;">🔒 Everything was done locally - your data never left your computer</p>
            </div>
            """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div class="info-card" style="text-align: center; margin-top: 3rem;">
    <h3 style="margin-top: 0;">🔒 Privacy & Security</h3>
    <p>This application runs completely offline using your local DeepSeek model.</p>
    <p style="margin-bottom: 0.5rem;"><strong>✓ No data sent to external servers</strong></p>
    <p style="margin-bottom: 0.5rem;"><strong>✓ All processing happens on your machine</strong></p>
    <p style="margin-bottom: 0;"><strong>✓ Your documents remain completely private</strong></p>
    <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
    <p style="margin: 0; opacity: 0.8;">⏰ Session: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
