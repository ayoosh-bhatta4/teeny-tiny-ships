"""
test_everything.py
------------------
This script proves that our modular .py files work perfectly 
by importing them and running them through the evaluator.
"""

from evaluator import run_evaluations
from inference_custom import organizer_custom
from inference_groq import organizer_groq

print("========================================")
print("  TESTING CUSTOM DISTILBERT MODEL       ")
print("========================================")
run_evaluations(organizer_custom, system_name="Custom HF Model")

print("\n========================================")
print("  TESTING GROQ API MODEL                ")
print("========================================")
run_evaluations(organizer_groq, system_name="Groq Llama-3")