@echo off
echo Starting OPTIMIZED DeepSeek PDF Processor...
echo.
echo Performance improvements:
echo - 10-20x faster processing
echo - Greedy decoding instead of sampling
echo - Reduced token generation (100 vs 512)
echo - Smaller input size (800 vs 1500 chars)
echo - Optional table extraction
echo.
streamlit run app_optimized.py --server.headless true --server.port 8501
pause
