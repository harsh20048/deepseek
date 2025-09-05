# DeepSeek PDF Processor - Automated GitHub Setup
# This script automates the GitHub repository creation and push process

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "           DeepSeek PDF Processor - Automated GitHub Setup" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Git status
Write-Host "[1/6] Checking Git status..." -ForegroundColor Yellow
git status
Write-Host ""

# Step 2: Show current files
Write-Host "[2/6] Current repository files:" -ForegroundColor Yellow
Get-ChildItem -Name | ForEach-Object { Write-Host "  ✅ $_" -ForegroundColor Green }
Write-Host ""

# Step 3: Ensure all files are committed
Write-Host "[3/6] Ensuring all files are committed..." -ForegroundColor Yellow
git add .
git commit -m "Final commit before GitHub push" 2>$null
Write-Host ""

# Step 4: Open GitHub
Write-Host "[4/6] Opening GitHub repository creation page..." -ForegroundColor Yellow
Write-Host "Opening https://github.com/new in your default browser..." -ForegroundColor Cyan
Start-Process "https://github.com/new"
Write-Host ""

# Step 5: Instructions
Write-Host "[5/6] Repository Creation Instructions:" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Yellow
Write-Host "1. Repository name: deepseek-pdf-processor" -ForegroundColor White
Write-Host "2. Description: A privacy-first PDF processing application using local DeepSeek model" -ForegroundColor White
Write-Host "3. Choose Public or Private" -ForegroundColor White
Write-Host "4. DO NOT check any boxes (README, .gitignore, license)" -ForegroundColor White
Write-Host "5. Click 'Create repository'" -ForegroundColor White
Write-Host ""

# Step 6: Get username and push
Write-Host "[6/6] Ready to push to GitHub..." -ForegroundColor Yellow
$username = Read-Host "Enter your GitHub username"

if ($username) {
    Write-Host ""
    Write-Host "Adding remote origin..." -ForegroundColor Cyan
    git remote add origin "https://github.com/$username/deepseek-pdf-processor.git"
    
    Write-Host "Renaming branch to main..." -ForegroundColor Cyan
    git branch -M main
    
    Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
    git push -u origin main
    
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "                    SUCCESS! " -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your repository is now live at:" -ForegroundColor White
    Write-Host "https://github.com/$username/deepseek-pdf-processor" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "You can now:" -ForegroundColor White
    Write-Host "- Share your repository with others" -ForegroundColor Gray
    Write-Host "- Clone it on other machines" -ForegroundColor Gray
    Write-Host "- Continue development with version control" -ForegroundColor Gray
} else {
    Write-Host "No username provided. Exiting..." -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
Read-Host
