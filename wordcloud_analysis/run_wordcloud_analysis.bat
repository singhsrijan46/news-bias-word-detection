@echo off
REM ==============================================================================
REM Run Biased Word Cloud & Model Intrinsic Bias Analysis
REM Generates 12 Word Clouds, Frequency Comparisons, and 5 Bias Audit Plots
REM ==============================================================================

echo [INFO] Starting Biased Word Cloud & Model Self-Bias Analysis...
cd /d "%~dp0\.."
python wordcloud_analysis\wordcloud_comparator.py

echo.
echo [SUCCESS] Word cloud analysis finished!
echo [INFO] Check wordcloud_analysis\MODEL_BIAS_AND_WORDCLOUD_REPORT.md and wordcloud_analysis\plots\
pause
