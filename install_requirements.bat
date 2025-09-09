@echo off
echo Installing DeepSeek PDF Processor Requirements...
echo.

echo Installing basic requirements...
pip install -r requirements.txt

echo.
echo Installing accelerate for GPU support...
pip install accelerate

echo.
echo Installation complete!
echo.
echo You can now run the PDF processor with:
echo   python test.py
echo.
pause
