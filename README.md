# mediscan — AI-Powered Medicine Identification

A modern web application that identifies medicines from package images using OCR, barcode scanning, and intelligent matching. Built with Flask, Tesseract OCR, and Python.

## Features

- 📸 **Image Upload & OCR**: Upload medicine package images with automatic text extraction
- 📊 **Barcode/QR Scanning**: First-class barcode and QR code detection (pyzbar + OpenCV)
- ✂️ **Crop-to-Heading Tool**: Interactive client-side cropping to focus on brand name areas
- 🎯 **Confidence Scoring**: Compares detected heading with matched medicine name
- 🔍 **Fuzzy Matching**: Advanced string matching using RapidFuzz for best results
- 📋 **Top Candidates**: Shows up to 5 similar medicines with match scores
- 👤 **Optional Authentication**: Sign up/login to enable future history features
- 🎨 **Modern UI**: Polished, responsive design with loading states

## Prerequisites

- **Python 3.8+**
- **Tesseract OCR** (required for text extraction)
- **Medicine.csv** file in the project root

## Installation & Setup

### 1. Clone/Navigate to Project
```bash
cd "C:\Users\PEETER K VARKEY\OneDrive\Desktop\trial"
```

### 2. Create Virtual Environment (Recommended)
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate

# Or use venv if available
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- Flask 3.0.3
- Pillow (image processing)
- pytesseract (OCR wrapper)
- rapidfuzz (fuzzy string matching)
- opencv-python (image processing & QR detection)
- pyzbar (barcode/QR decoding)

### 4. Install Tesseract OCR (Windows)

**Option A: Automatic (Recommended)**
- Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default path: `C:\Program Files\Tesseract-OCR\`
- The app will auto-detect Tesseract in common paths

**Option B: Manual PATH**
- Install Tesseract anywhere and add `tesseract.exe` to your system PATH

**Note for Barcode Support:**
- For full barcode support (non-QR), you may need ZBar libraries
- QR codes work without ZBar (uses OpenCV fallback)

### 5. Ensure Medicine.csv Exists
- Place `Medicine.csv` in the project root with columns:
  - Medicine Name, Composition, Uses, Side_effects, Image URL, Manufacturer
  - Excellent Review %, Average Review %, Poor Review %

## Running the Application

### Start the Server
```bash
python app.py
```

The app will:
1. **First run only**: Create `medicine.db` SQLite database from `Medicine.csv`
2. Create `users` table (for authentication)
3. Create `uploads/` directory for uploaded images
4. Start Flask server on `http://localhost:5000`

### Access the App
- Open your browser: **http://localhost:5000**
- Or: **http://127.0.0.1:5000**

### Routes
- `/` - Scanning interface (upload image or search by name)
- `/home` - Landing page with app overview
- `/signup` - Create account (optional)
- `/login` - Log in (optional)
- `/logout` - Log out

## Usage Guide

### 1. Scan Medicine Package
1. Click **"Scan Medicine"** in navigation or go to `/`
2. Click **"Choose File"** and select a medicine package image
3. **Optional**: Drag to crop the brand name area for better accuracy
4. Click **"Scan & Find"**
5. View results with confidence scores and medicine details

### 2. Search by Name
1. Type medicine name in the search box
2. Click **"Search"**
3. Results show best match and candidates

### 3. Understanding Results
- **Best Match**: Highest confidence match from database
- **Confidence Score**: Percentage comparing heading to medicine name
- **Top Candidates**: Alternative matches with similarity scores
- **Identified via barcode**: Shows when barcode/QR was successfully decoded

## Project Structure

```
trial/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Medicine.csv          # Medicine database (CSV)
├── medicine.db           # SQLite database (auto-created)
├── uploads/              # Uploaded images directory
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── home.html         # Landing page
│   ├── index.html        # Scanning/search interface
│   ├── result.html       # Results display
│   ├── signup.html       # Sign up page
│   └── login.html        # Login page
└── static/
    ├── style.css         # Stylesheet
    ├── mediscan-logo-alt.svg
    └── mediscan-icon-alt.svg
```

## Troubleshooting

### OCR Issues
- **"OCR failed"**: 
  - Verify Tesseract is installed: `tesseract --version`
  - Check that Tesseract is in PATH or common Windows paths
  - Ensure image file is readable (PNG, JPG, JPEG, WEBP)

### Poor Text Extraction
- Use higher resolution images (at least 800px width recommended)
- Avoid blur, glare, or shadows
- Crop to brand name area using the built-in crop tool
- Ensure text is horizontal and clearly visible

### No Good Match Found
- Try using the manual search box with exact medicine name
- Check that the medicine exists in `Medicine.csv`
- Verify image contains readable text

### Barcode Not Detected
- Ensure barcode/QR is clearly visible and not damaged
- Try different image orientations
- QR codes work via OpenCV fallback even without ZBar
- For non-QR barcodes, you may need ZBar libraries

### Database Issues
- Delete `medicine.db` to reimport from `Medicine.csv`
- Ensure CSV file is in correct format with expected columns

### Port Already in Use
- Change port in `app.py`: `app.run(..., port=5001)`
- Or kill process using port 5000

## Development

### Running in Debug Mode
Already enabled in `app.py`:
```python
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
```

### Custom Port
Set environment variable:
```powershell
$env:PORT=8000
python app.py
```

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Database**: SQLite3
- **OCR**: Tesseract OCR via pytesseract
- **Image Processing**: Pillow (PIL), OpenCV
- **Barcode**: pyzbar, OpenCV QRCodeDetector
- **Matching**: RapidFuzz (fuzzy string matching)
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)

## License

This project is for demonstration/educational purposes.

---

**Need help?** Check that all dependencies are installed and Tesseract OCR is properly configured.
