"""
inference_groq.py
-----------------
This module handles inference using the Groq API (Llama-3 70B) 
by forcing it to return a strictly formatted JSON object.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load secrets and initialize client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# The strict formatting prompt for the LLM
system_prompt = """
TASK:
For each item in the input, output its category and amount.
Categories: Food, Transport, Utilities, Entertainment, Shopping, Health, Home.

RULES:
- Output ONLY valid JSON, nothing else.
- No explanation, no extra text.
- Format MUST be exactly: {"item_name": {"category": "CategoryName", "amount": 100}}
- Amount must be a number.
- Category must be a string from the list above.
"""

# The Organizer Function
def organizer_groq(in_str):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": in_str}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0, # Temperature 0 for maximum consistency during inference
            response_format={"type": "json_object"}
        )
        
        raw_json_string = chat_completion.choices[0].message.content
        parsed_items = json.loads(raw_json_string)
        
        final_totals = {}
        
        # Extract the categories and amounts from the nested JSON
        for item, details in parsed_items.items():
            cat = details.get("category", "Unknown")
            amt = details.get("amount", 0)
            final_totals[cat] = final_totals.get(cat, 0) + amt
            
        return final_totals
        
    except Exception as e:
        print(f"⚠️ Groq API Error: {e}")
        return {}

# Quick test block that only runs if you execute this specific file
if __name__ == "__main__":
    print("\n--- Testing Groq Inference ---")
    test_input = "swiggy 500 uber 250"
    print(f"Input: {test_input}")
    print(f"Output: {organizer_groq(test_input)}")