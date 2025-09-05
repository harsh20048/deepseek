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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
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
    """Extract text from uploaded PDF"""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            total_pages = len(pdf.pages)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                
                progress_bar.progress(i / total_pages)
                status_text.text(f"Processing page {i}/{total_pages}")
            
            progress_bar.empty()
            status_text.empty()
            
            return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

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

# Main UI
st.markdown('<h1 class="main-header">🔍 DeepSeek PDF Processor</h1>', unsafe_allow_html=True)
st.markdown("**Process German PDFs with your local DeepSeek model - 100% Offline & Private**")

# Sidebar for model status
with st.sidebar:
    st.header("🤖 Model Status")
    
    if not st.session_state.model_loaded:
        if st.button("Load DeepSeek Model", type="primary"):
            if load_model():
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model")
    else:
        st.success("✅ Model loaded and ready!")
        st.info("🔒 Running completely offline")
    
    st.header("📋 Instructions")
    st.markdown("""
    1. **Load Model**: Click the button above to load your local DeepSeek model
    2. **Upload PDF**: Upload a German PDF file using the file uploader
    3. **Process**: Click "Process PDF" to extract structured data
    4. **View Results**: See extracted fields and table data below
    """)

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
                        
                        # Field extraction query
                        field_query = """
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
                        
                        with st.spinner("🤖 Extracting fields with DeepSeek..."):
                            # Truncate PDF text if too long
                            max_pdf_length = 1500  # Limit PDF text length
                            truncated_pdf = pdf_text[:max_pdf_length] + "..." if len(pdf_text) > max_pdf_length else pdf_text
                            
                            full_prompt = f"PDF Content:\n{truncated_pdf}\n\n{field_query}"
                            fields_response = query_local_model(full_prompt, max_new_tokens=256)
                        
                        # Try to parse as JSON, fallback to raw text
                        try:
                            parsed_fields = json.loads(fields_response)
                        except:
                            parsed_fields = {"Raw_Response": fields_response}
                        
                        # Store results
                        file_result = {
                            "filename": uploaded_file.name,
                            "fields": parsed_fields,
                            "raw_response": fields_response,
                            "text_length": len(pdf_text)
                        }
                        all_results.append(file_result)
                        
                        # Display fields for this file
                        with st.expander(f"📋 Fields from {uploaded_file.name}", expanded=True):
                            st.json(parsed_fields)
                        
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
                                st.dataframe(df, use_container_width=True)
                            
                            # Store table data
                            all_tables.append(df)
                        else:
                            st.warning(f"No table data found in {uploaded_file.name}")
            
            # Summary section
            st.markdown("---")
            st.header("📊 Batch Processing Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Processed", len(all_results))
            with col2:
                total_tables = sum(len(table) for table in all_tables)
                st.metric("Total Table Rows", total_tables)
            with col3:
                total_chars = sum(result['text_length'] for result in all_results)
                st.metric("Total Characters", f"{total_chars:,}")
            
            # Combined results
            if all_results:
                st.subheader("📋 All Extracted Fields")
                for result in all_results:
                    with st.expander(f"Fields from {result['filename']}", expanded=False):
                        st.json(result['fields'])
            
            # Combined tables
            if all_tables:
                st.subheader("📊 All Table Data Combined")
                combined_df = pd.concat(all_tables, ignore_index=True)
                st.dataframe(combined_df, use_container_width=True)
                
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
            
            # Final success message
            st.success("🎉 Batch processing completed successfully!")
            st.info("🔒 All processing was done locally - no data was sent to external servers")

# Footer
st.markdown("---")
st.markdown("**🔒 Privacy First**: This application runs completely offline using your local DeepSeek model. No data is sent to external servers.")
st.markdown(f"**⏰ Last updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
