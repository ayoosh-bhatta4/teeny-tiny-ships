"""
evaluator.py
------------
A universal testing engine. It accepts any organizer function as an argument,
feeds it the test cases, and calculates the score.
"""

import os
import json

def run_evaluations(organizer_func, system_name="Model"):
    print(f"\n🚀 Booting up the Evaluation Suite for: {system_name}")

    # 1. Load the Test Cases
    test_file = "test_cases.json"
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Could not find {test_file}. Make sure it is in the root folder.")
    
    with open(test_file, 'r') as f:
        test_cases = json.load(f)

    # 2. The Universal Evaluation Loop
    print("-" * 50)
    passed = 0
    total = len(test_cases)

    for test in test_cases:
        print(f"Running Test: {test['test_name']}")
        
        try:
            # Pass the input string to WHATEVER function was provided
            extracted_categories = organizer_func(test['input'])
            
            # Calculate the total sum of all extracted categories
            calculated_sum = sum(extracted_categories.values())
            
            if calculated_sum == test['expected_sum']:
                print(f"  ✅ PASS: Got {calculated_sum}")
                passed += 1
            else:
                print(f"  ❌ FAIL: Expected {test['expected_sum']}, got {calculated_sum}")
                print(f"     Output: {extracted_categories}")
                
        except Exception as e:
             print(f"  ⚠️ ERROR: Function crashed on this test: {e}")

    print(f"\n--- {system_name} Evals Complete: {passed}/{total} Passed ---\n")