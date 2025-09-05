@echo off
title DeepSeek PDF Processor - Automated GitHub Setup
color 0A

echo.
echo ================================================================
echo           DeepSeek PDF Processor - Automated GitHub Setup
echo ================================================================
echo.

echo [1/6] Checking Git status...
git status
echo.

echo [2/6] Checking current repository files...
dir /b
echo.

echo [3/6] Ensuring all files are committed...
git add .
git commit -m "Final commit before GitHub push" 2>nul
echo.

echo [4/6] Opening GitHub repository creation page...
echo Opening https://github.com/new in your default browser...
start https://github.com/new
echo.

echo [5/6] Waiting for you to create the repository...
echo.
echo INSTRUCTIONS:
echo =============
echo 1. Repository name: deepseek-pdf-processor
echo 2. Description: A privacy-first PDF processing application using local DeepSeek model
echo 3. Choose Public or Private
echo 4. DO NOT check any boxes (README, .gitignore, license)
echo 5. Click "Create repository"
echo.

set /p username="Enter your GitHub username: "

echo.
echo [6/6] Pushing code to GitHub...
echo.

echo Adding remote origin...
git remote add origin https://github.com/%username%/deepseek-pdf-processor.git

echo Renaming branch to main...
git branch -M main

echo Pushing to GitHub...
git push -u origin main

echo.
echo ================================================================
echo                    SUCCESS! 
echo ================================================================
echo.
echo Your repository is now live at:
echo https://github.com/%username%/deepseek-pdf-processor
echo.
echo You can now:
echo - Share your repository with others
echo - Clone it on other machines
echo - Continue development with version control
echo.
echo Press any key to exit...
pause >nul
