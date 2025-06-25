##Handles text extraction from PDF and image

import fitz  #PyMuPDF libraby for extracting text from pdf
import pytesseract  #Tesseract OCR for extracting text from images and scanned pdfs
from PIL import Image
import os

#path to tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#function to extract text from uploaded pdf
def extract_text_from_pdf(pdf_path):
    text=""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text+=page.get_text()
    return text.strip()


#function to extract text from uploaded images
def extract_text_from_image(image_path):
    try:
        image=Image.open(image_path)
        text=pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error reading image: {e}"
