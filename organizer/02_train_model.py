"""
02_train_model.py
-----------------
This script takes the synthetic dataset and fine-tunes a DistilBERT 
model to classify personal finance transactions.

Note: This model was already trained and pushed to the Hugging Face Hub. 
This script is kept for reproducibility and future retraining.
"""

import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# Load environment variables (Not strictly needed for local training, but good practice if pushing to hub later)
load_dotenv()

def train_expense_classifier():
    print("🚀 Starting Model Training Pipeline...")

    # 1. Load the dataset
    data_path = "finance_training_data_1k.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}. Please run 01_generate_data.py first.")
    
    df = pd.read_csv(data_path)
    
    # 2. Create Label Mappings (Text to Math)
    categories = df['label'].unique().tolist()
    label2id = {category: idx for idx, category in enumerate(categories)}
    id2label = {idx: category for idx, category in enumerate(categories)}
    
    # Convert string labels to integer IDs for the neural network
    df['label'] = df['label'].map(label2id)
    
    # Convert Pandas DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df)
    
    # Split into Train (80%) and Test (20%)
    dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
    
    # 3. Load the pre-trained Brain and Tokenizer
    model_name = "distilbert-base-uncased"
    print(f"Loading base model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id
    )
    
    # 4. Tokenization Function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset_split.map(tokenize_function, batched=True)
    
    # 5. Training Setup
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
    )
    
    # 6. Run Training
    print("Commencing Training...")
    trainer.train()
    
    # 7. Save the finalized model locally
    output_model_dir = "./final_tiny_model"
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"✅ Training Complete! Model saved to {output_model_dir}")

if __name__ == "__main__":
    # Commented out to prevent accidental 10-minute training runs!
    # train_expense_classifier()
    print("Training script configured. Uncomment the function call at the bottom to trigger a new training run.")