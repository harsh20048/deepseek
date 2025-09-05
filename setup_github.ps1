# GitHub Repository Setup Script
# This script will help you create and push to GitHub

Write-Host "🚀 DeepSeek PDF Processor - GitHub Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "✅ Git is available: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Check current git status
Write-Host "`n📋 Checking current repository status..." -ForegroundColor Yellow
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "⚠️  There are uncommitted changes. Committing them first..." -ForegroundColor Yellow
    git add .
    git commit -m "Update project files before GitHub push"
}

# Check if we're in a git repository
$gitDir = git rev-parse --git-dir 2>$null
if (-not $gitDir) {
    Write-Host "❌ Not in a git repository. Initializing..." -ForegroundColor Red
    git init
    git add .
    git commit -m "Initial commit: DeepSeek PDF Processor"
}

Write-Host "`n📁 Current repository status:" -ForegroundColor Yellow
git log --oneline -5

Write-Host "`n🔗 Next steps to create GitHub repository:" -ForegroundColor Cyan
Write-Host "1. Go to https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: deepseek-pdf-processor" -ForegroundColor White
Write-Host "3. Description: A privacy-first PDF processing application using local DeepSeek model" -ForegroundColor White
Write-Host "4. Choose Public or Private" -ForegroundColor White
Write-Host "5. DO NOT initialize with README, .gitignore, or license (we already have them)" -ForegroundColor White
Write-Host "6. Click 'Create repository'" -ForegroundColor White

Write-Host "`n📝 After creating the repository, run these commands:" -ForegroundColor Cyan
Write-Host "git remote add origin https://github.com/YOUR_USERNAME/deepseek-pdf-processor.git" -ForegroundColor White
Write-Host "git branch -M main" -ForegroundColor White
Write-Host "git push -u origin main" -ForegroundColor White

Write-Host "`n💡 Replace YOUR_USERNAME with your actual GitHub username!" -ForegroundColor Yellow

Write-Host "`n📊 Project files ready for GitHub:" -ForegroundColor Green
Get-ChildItem -Name | ForEach-Object { Write-Host "  ✅ $_" -ForegroundColor Green }

Write-Host "`n🎉 Your project is ready to be pushed to GitHub!" -ForegroundColor Green
Write-Host "Press any key to continue..." -ForegroundColor Gray
Read-Host
