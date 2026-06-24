@echo off
REM Overnight flagship run: train -> evaluate -> model card. Logs to overnight_run.log.
set PYTHONUNBUFFERED=1
cd /d C:\Users\sharv\projects\kidney-tumor-ct-segmentation
set PY=C:\Users\sharv\.venvs\cv\Scripts\python.exe
set MLFLOW_TRACKING_URI=file:///C:/Users/sharv/projects/kidney-tumor-ct-segmentation/mlruns

echo [%date% %time%] START train >> overnight_run.log
"%PY%" src\train.py --cache-dir "C:/Users/sharv/data/c4kc-kits-cache" --epochs 200 --val-interval 10 >> overnight_run.log 2>&1

echo [%date% %time%] START evaluate >> overnight_run.log
"%PY%" src\evaluate.py >> overnight_run.log 2>&1

echo [%date% %time%] START model card >> overnight_run.log
"%PY%" src\make_model_card.py >> overnight_run.log 2>&1

echo [%date% %time%] DONE >> overnight_run.log
