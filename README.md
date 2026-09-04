# Marine Gauge AI

Starter project for extracting marine gauge information from uploaded images.

## Planned pipeline
1. Image preprocessing (OpenCV)
2. Gauge detection (YOLO)
3. Gauge type classification (analog/digital)
4. Digital reading (PaddleOCR)
5. Analog reading (OpenCV/custom algorithm)
6. Context extraction (Vision Language Model)
7. Validation
8. Structured JSON output

## Setup

Create a virtual environment:

Windows:
    python -m venv .venv
    .venv\Scripts\activate

Install dependencies:
    pip install -r requirements.txt

Run the starter pipeline:
    python src/main.py

The current files are intentionally starter implementations. We will replace each placeholder with the actual model and logic using your company's gauge images.
