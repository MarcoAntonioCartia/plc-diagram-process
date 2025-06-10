@echo off
echo.
echo ===============================================
echo   PLC Diagram Processor Environment
echo ===============================================
echo.
echo Activating virtual environment...
call "D:\MarMe\github\0.3\plc-diagram-processor\yolovenv\Scripts\activate.bat"
echo.
echo Environment activated!
echo Data directory: D:\MarMe\github\0.3\plc-data
echo Python: D:\MarMe\github\0.3\plc-diagram-processor\yolovenv\Scripts\python.exe
echo.
echo Available commands:
echo   python src/detection/run_complete_pipeline.py
echo   python src/ocr/run_text_extraction.py
echo   python scripts/manage_datasets.py
echo   python scripts/manage_models.py
echo.
cmd /k
