# Medicine OCR Lookup (Flask + SQLite)

Simple web app that scans a medicine package image, performs OCR to read text, then finds the closest matching medicine in a SQLite database imported from `Medicine.csv` and displays details.

## Setup

### 1) Python environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

### 2) Install Tesseract OCR (Windows)
- Download installer: `https://github.com/UB-Mannheim/tesseract/wiki`
- Install to the default path: `C:\\Program Files\\Tesseract-OCR\\`
- Ensure `tesseract.exe` is in PATH, or the app will auto-detect common paths.

### 3) First run (imports CSV → SQLite)
The first run creates `medicine.db` from `Medicine.csv` (must be in the same folder).
```bash
python app.py
```
Open: `http://localhost:5000`

## Usage
- Upload a clear photo of the package focusing on the brand name.
- The app extracts text with OCR and fuzzy-matches against the database.
- Results show the best match and top candidates.

## Notes
- If `rapidfuzz` is unavailable, the app falls back to a simpler matching strategy.
- CSV columns expected:
  - Medicine Name, Composition, Uses, Side_effects, Image URL, Manufacturer, Excellent Review %, Average Review %, Poor Review %
 - OCR preprocessing tries multiple filters and Tesseract modes automatically.

## Troubleshooting
- "OCR failed": Ensure Tesseract is installed and accessible.
- No good match: Try a sharper image or crop to the medicine name area.
 - If extraction is poor:
   - Ensure image is not too small; higher resolution helps.
   - Avoid glare/shadows; keep text horizontal when possible.
   - Try the manual search box if the brand name is known.
