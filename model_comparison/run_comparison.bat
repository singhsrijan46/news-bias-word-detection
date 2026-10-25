@echo off
REM ==============================================================================
REM Run Top 10 Research Paper Models Comparison
REM Generates Accuracy Leaderboard, Word Counts, Confusion Matrices, and Graphs
REM ==============================================================================

echo [INFO] Starting Top 10 Model Comparison Benchmark...
cd /d "%~dp0\.."
python model_comparison\compare_models.py

echo.
echo [SUCCESS] Model comparison finished! Check model_comparison\MODEL_COMPARISON_REPORT.md and model_comparison\plots\
pause
