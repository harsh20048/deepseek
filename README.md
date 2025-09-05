# 🔍 DeepSeek PDF Processor

A powerful, privacy-first PDF processing application that uses your local DeepSeek model to extract structured data from German PDFs. Process multiple documents simultaneously with 100% offline operation.

## ✨ Features

- **🔒 100% Offline & Private** - Uses your local DeepSeek model, no external API calls
- **📄 Multi-Document Processing** - Upload and process multiple PDFs simultaneously
- **🤖 AI-Powered Extraction** - Extracts structured fields and table data using DeepSeek
- **📊 Beautiful Web UI** - Modern Streamlit interface with real-time progress
- **💾 Export Options** - Download results as CSV or JSON
- **🇩🇪 German Language Support** - Optimized for German quotation documents
- **⚡ Batch Processing** - Process multiple files with comprehensive results summary

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- DeepSeek model files (see Model Setup below)
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepseek-pdf-processor.git
   cd deepseek-pdf-processor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your DeepSeek model**
   - Download DeepSeek model files to the `model/` directory
   - Ensure all required files are present (see Model Setup section)

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
   Or use the batch file:
   ```bash
   run_app.bat
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Load your DeepSeek model
   - Upload and process your German PDFs!

## 📁 Project Structure

```
deepseek-pdf-processor/
├── app.py                 # Main Streamlit application
├── test.py               # Command-line PDF processor
├── requirements.txt      # Python dependencies
├── run_app.bat          # Windows launcher script
├── README.md            # This file
├── model/               # DeepSeek model files (add your model here)
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   └── pytorch_model.bin
└── db/                  # Chroma database (created automatically)
    └── chroma.sqlite3
```

## 🤖 Model Setup

### Required Model Files

Place your DeepSeek model files in the `model/` directory:

- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer data
- `tokenizer_config.json` - Tokenizer configuration
- `generation_config.json` - Generation settings
- `model.safetensors` - Model weights (safetensors format)
- `pytorch_model.bin` - Model weights (PyTorch format)

### Model Path Configuration

Update the model path in `app.py` if your model is located elsewhere:

```python
model_path = "C:/Users/HARSH/OneDrive/Desktop/pdf/model"  # Update this path
```

## 📖 Usage

### Web Interface (Recommended)

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Load your model**
   - Click "Load DeepSeek Model" in the sidebar
   - Wait for the model to load (first time may take a few minutes)

3. **Upload PDFs**
   - Select one or more German PDF files
   - Click "Process All PDFs"

4. **View results**
   - Extracted fields in JSON format
   - Table data in interactive tables
   - Download results as CSV or JSON

### Command Line Interface

For batch processing without the web interface:

```bash
python test.py
```

## 🔧 Configuration

### Environment Variables

The application sets these environment variables for offline operation:

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = "dummy_token"
os.environ["OPENAI_API_KEY"] = "dummy_key"
```

### Model Parameters

Adjust model parameters in `app.py`:

```python
# Text truncation limit
max_pdf_length = 1500

# Model generation parameters
max_new_tokens = 512
temperature = 0.7
```

## 📊 Extracted Data

### Structured Fields

- **Date (Datum)** - Document date
- **Angebot** - Quotation number
- **SenderCompany** - Company name
- **SenderAddress** - Complete address

### Table Data

- **Artikel-Nr** - Article number
- **Bezeichnung** - Description
- **Menge** - Quantity
- **Preis** - Unit price
- **Gesamt** - Total price

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure all model files are in the correct directory
   - Check file permissions
   - Verify model path in configuration

2. **Memory Issues**
   - Reduce `max_pdf_length` for large documents
   - Process fewer files simultaneously
   - Close other applications to free memory

3. **Token Length Errors**
   - The app automatically truncates long PDFs
   - Adjust `max_input_length` if needed

### Performance Tips

- **First Run**: Model loading takes time on first use
- **Memory**: Ensure sufficient RAM for your model size
- **Batch Processing**: Process files in smaller batches for large datasets

## 🔒 Privacy & Security

- **100% Offline**: No data sent to external servers
- **Local Processing**: All AI processing happens on your machine
- **No Data Collection**: No usage analytics or data logging
- **Secure**: Your documents never leave your computer

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section
2. Review the GitHub issues
3. Create a new issue with detailed information

## 🙏 Acknowledgments

- [DeepSeek](https://github.com/deepseek-ai) for the amazing language model
- [Streamlit](https://streamlit.io/) for the web framework
- [Transformers](https://huggingface.co/transformers/) for model integration
- [PDFPlumber](https://github.com/jsvine/pdfplumber) for PDF text extraction

## 📈 Roadmap

- [ ] Support for more document types
- [ ] Advanced table extraction algorithms
- [ ] Custom field extraction templates
- [ ] API endpoint for integration
- [ ] Docker containerization
- [ ] Multi-language support

---

**Made with ❤️ for privacy-conscious document processing**
