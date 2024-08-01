import easyocr
import cv2
from PIL import Image
from spellchecker import SpellChecker
import torch
from transformers import BertTokenizer, BertForMaskedLM

reader = easyocr.Reader(['ar'])

spell = SpellChecker(language='ar')

model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def correct_spelling(text):
    corrected_text = []
    for word in text.split():
        corrected_word = spell.correction(word)
        corrected_text.append(corrected_word)
    return ' '.join(corrected_text)

def correct_with_bert(text):
    tokens = tokenizer.tokenize(text)
    masked_tokens = ['[MASK]' if spell.unknown([token]) else token for token in tokens]

    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_ids)
    predictions = outputs.logits

    predicted_tokens = []
    for i, token in enumerate(masked_tokens):
        if token == '[MASK]':
            predicted_id = torch.argmax(predictions[0, i]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_id])[0]
            predicted_tokens.append(predicted_token)
        else:
            predicted_tokens.append(token)
    
    corrected_text = tokenizer.convert_tokens_to_string(predicted_tokens)
    return corrected_text

# Read the image
image_path = "/mnt/d/valify/0072.jpg"
image = cv2.imread(image_path)

# Perform OCR
results = reader.readtext(image)

# Print the results with spell and language model correction
for (bbox, text, prob) in results:
    # Apply spell check
    corrected_text_spell = correct_spelling(text)
    
    # Apply language model correction
    corrected_text_bert = correct_with_bert(corrected_text_spell)
    
    print(f"Detected text: {text} (Confidence: {prob})")
    print(f"Corrected text (Spell Check): {corrected_text_spell}")
    print(f"Corrected text (BERT): {corrected_text_bert}")