"""
inference_custom.py
-------------------
This module handles inference using our fine-tuned DistilBERT model
hosted on the Hugging Face Hub, combined with our Regex pre-processor.
"""

import os
import re
from dotenv import load_dotenv
from transformers import pipeline

# Load secrets (HF_TOKEN) in case your repo is private
load_dotenv()

# 1. Initialize the pipeline ONCE when the script loads
MODEL_ID = "ayoosh-bhatta4/teeny-finance-classifier" # Change if your HF repo is named differently
print(f"Loading custom model '{MODEL_ID}' from Hugging Face...")

expense_classifier = pipeline(
    task="text-classification",
    model=MODEL_ID,
    tokenizer="distilbert-base-uncased"
)

# 2. The Organizer Function
def organizer_custom(in_str):
    # Normalize and split by common separators or numbers followed by letters
    normalized_str = re.sub(r'(?i)\band\b|\||,', ' [SPLIT] ', in_str)
    normalized_str = re.sub(r'(\d+)\s+(?=[a-zA-Z])', r'\1 [SPLIT] ', normalized_str)
    
    raw_parts = [part.strip() for part in normalized_str.split('[SPLIT]') if part.strip()]
    final_totals = {}
    
    for part in raw_parts:
        amounts = re.findall(r'\d+', part)
        if amounts:
            amount = int(amounts[-1])
            clean_text = re.sub(r'\d+', '', part).strip()
            
            if clean_text:
                prediction = expense_classifier(clean_text)[0]
                cat = prediction['label']
                final_totals[cat] = final_totals.get(cat, 0) + amount
                
    return final_totals

# Quick test block that only runs if you execute this specific file
if __name__ == "__main__":
    print("\n--- Testing Custom Inference ---")
    test_input = "swiggy 500 uber 250"
    print(f"Input: {test_input}")
    print(f"Output: {organizer_custom(test_input)}")