from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import easyocr
import cv2
from spellchecker import SpellChecker
import torch
import numpy as np

app = FastAPI()

reader = easyocr.Reader(['ar'])

spell = SpellChecker(language='ar')

# Load the YOLO model
model_path = "./models/best.pt"
yolo_model = YOLO(model_path)

# Function to correct spelling
def correct_spelling(text):
    corrected_text = []
    for word in text.split():
        corrected_word = spell.correction(word)
        corrected_text.append(corrected_word)
    return ' '.join(corrected_text)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform inference using YOLO
    results = yolo_model.predict(source=image)

    final_texts = []

    for det in results[0].boxes:
        x1, y1, x2, y2 = det.xyxy[0]
        conf, cls = det.conf, det.cls
        cls = int(cls)
        if cls == 0:
            roi = image[int(y1):int(y2), int(x1):int(x2)]

            # Perform OCR on the ROI
            ocr_results = reader.readtext(roi)
            for (bbox, text, prob) in ocr_results:
                corrected_text_spell = correct_spelling(text)
                
                final_texts.append({
                    # "detected_text": text,
                    # "confidence": prob,
                    "corrected_text_spell": corrected_text_spell,
                })

    return JSONResponse(content=final_texts)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
