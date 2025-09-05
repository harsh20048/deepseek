@echo off
echo.
echo ========================================
echo  DeepSeek PDF Processor - GitHub Setup
echo ========================================
echo.

echo Checking Git status...
git status
echo.

echo Current repository files:
dir /b
echo.

echo ========================================
echo  NEXT STEPS TO CREATE GITHUB REPOSITORY:
echo ========================================
echo.
echo 1. Go to: https://github.com/new
echo.
echo 2. Repository name: deepseek-pdf-processor
echo.
echo 3. Description: A privacy-first PDF processing application using local DeepSeek model
echo.
echo 4. Choose Public or Private
echo.
echo 5. DO NOT check any of these boxes:
echo    - Add a README file
echo    - Add .gitignore
echo    - Choose a license
echo    (We already have all of these)
echo.
echo 6. Click "Create repository"
echo.
echo ========================================
echo  AFTER CREATING REPOSITORY, RUN THESE:
echo ========================================
echo.
echo git remote add origin https://github.com/YOUR_USERNAME/deepseek-pdf-processor.git
echo git branch -M main
echo git push -u origin main
echo.
echo Replace YOUR_USERNAME with your actual GitHub username!
echo.
echo ========================================
echo  Your project is ready for GitHub!
echo ========================================
echo.
pause
