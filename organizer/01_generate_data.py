"""
01_generate_data.py
-------------------
This script uses the Groq API (Llama-3 70B) to generate a synthetic dataset 
of personal finance transactions. 

Note: This was run once to generate 'finance_training_data_1k.csv'. 
It is kept here for reproducibility and documentation of the pipeline.
"""

import os
import json
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load secrets from the .env file
load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_synthetic_data():
    all_training_examples = []
    total_batches = 10

    data_gen_prompt = """
    You are an expert synthetic data generator.
    Generate 100 realistic, highly varied examples of personal finance items (with prices) and map them to standard budgeting categories.
    Categories to use: Food, Transport, Utilities, Entertainment, Shopping, Health, Home.

    CRITICAL RULES:
    1. Output strictly valid JSON.
    2. The JSON must have a single root key called "dataset" containing a list of objects.
    3. Each object must have a "text" key (the user input) and a "label" key (the category).
    4. VARY THE FORMATS HEAVILY: Use slang, extreme typos, weird punctuation ("UBER-45"), long descriptions ("paid plumber 300 to fix sink"), and missing amounts.

    EXPECTED FORMAT:
    {
      "dataset": [
        {"text": "swiggy 200", "label": "Food"}
      ]
    }
    """

    print(f"🚀 Starting the Synthetic Data Factory...")

    for i in range(total_batches):
        try:
            print(f"Generating batch {i+1}/{total_batches}...")
            
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": data_gen_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.9, 
                response_format={"type": "json_object"}
            )
            
            raw_json = json.loads(response.choices[0].message.content)
            dataset_items = raw_json.get('dataset', [])
            
            # The Bouncer: Only accept perfectly formatted dictionaries
            valid_items = 0
            for item in dataset_items:
                if isinstance(item, dict) and 'text' in item and 'label' in item:
                    all_training_examples.append(item)
                    valid_items += 1
                    
            print(f"  -> Successfully extracted {valid_items} valid items.")
            
            # Pause to avoid hitting Groq's rate limits
            time.sleep(2)
            
        except Exception as e:
            print(f"⚠️ Error on batch {i+1}: {e}")

    # Convert to DataFrame and drop exact duplicates
    df = pd.DataFrame(all_training_examples)
    df = df.drop_duplicates(subset=['text'])
    
    # Save to CSV
    output_file = "finance_training_data_1k.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Success! Saved {len(df)} unique training examples to {output_file}")

if __name__ == "__main__":
    # We comment this out by default so accidentally running the file doesn't overwrite the existing CSV
    # generate_synthetic_data()
    print("Script is configured correctly. Uncomment the function call at the bottom to re-run generation.")