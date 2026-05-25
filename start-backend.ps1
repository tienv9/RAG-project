Set-Location "$PSScriptRoot\Backend FastAPI"
.\venv\Scripts\uvicorn main:app --reload --port 8000
