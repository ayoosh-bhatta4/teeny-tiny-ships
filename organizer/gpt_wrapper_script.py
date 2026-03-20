import requests
import json
import time

class InvalidOutputError(Exception):
    pass

usr_list = [] 
while(True):
    usr_in = input("Enter items of your list, type DONE to exit: ") 
    if(usr_in.lower() == "done"): 
        break 
    usr_list.append(usr_in) 
in_str = " ".join(usr_list)

prompt = f'''
TASK:
For each item in the input, output its category and amount.

RULES:
- Output ONLY valid JSON, nothing else
- No explanation, no extra text before or after the JSON
- Format MUST be exactly: {{"item_name": {{"category": "...", "amount": ...}}}}
- Amount must be a number
- Category must be a string

EXAMPLE INPUT:
swiggy 120 coffee 100 uber 300

EXAMPLE OUTPUT:
{{"swiggy": {{"category": "food", "amount": 120}}, "coffee": {{"category": "food", "amount": 100}}, "uber": {{"category": "transport", "amount": 300}}}}

YOUR INPUT:
{in_str}

YOUR OUTPUT (JSON only, starting with {{):
'''

MAX_RETRIES = 7
success = False
parsed = None

for attempt in range(MAX_RETRIES):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "seed": 42
                }
            },
            timeout=30  # phi can be slow, 5s is too tight
        )

        response.raise_for_status()
        data = response.json()
        raw_output = data["response"].strip()

        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1:
            raise InvalidOutputError("No JSON found")
        end += 1

        json_str = raw_output[start:end]
        json_str = json_str.replace("'", '"')
        parsed = json.loads(json_str)

        for key, value in parsed.items():
            if not isinstance(key, str):
                raise InvalidOutputError("Invalid key type")
            if not isinstance(value, dict):
                # print what the model actually returned to help debug
                print(f"DEBUG - raw model output: {raw_output}")
                print(f"DEBUG - parsed so far: {parsed}")
                raise InvalidOutputError(f"Expected dict for key '{key}', got {type(value).__name__}: {value}")

        success = True
        break

    except (requests.exceptions.RequestException,
            json.JSONDecodeError,
            KeyError,
            InvalidOutputError) as e:
        print(f"Attempt {attempt+1} failed: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(1)

if not success:
    print("Failed after all retries")
    exit()

# After parsing, replace the print(parsed) section with this:
totals = {}
for item, details in parsed.items():
    cat = details["category"]
    amt = int(details["amount"])
    totals[cat] = totals.get(cat, 0) + amt

print(totals)