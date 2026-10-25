@echo off
REM ==============================================================================
REM Run Optimized Champion Model (Highest Accuracy)
REM Trains the enhanced XGBoost architecture, evaluates on held-out test split,
REM generates comparison plots, and runs interactive bias prediction.
REM ==============================================================================

echo [INFO] Training & Evaluating Optimized News Bias Model...
cd /d "%~dp0\.."
python optimized_model\train_optimized_model.py

echo.
echo [INFO] Running Interactive News Bias Predictor Demonstration...
python optimized_model\predict_bias.py

echo.
echo [SUCCESS] Optimized Model finished! Check optimized_model\OPTIMIZED_MODEL_REPORT.md and plots\
pause
