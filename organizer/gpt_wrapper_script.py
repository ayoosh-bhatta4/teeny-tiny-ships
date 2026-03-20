import requests
import json
# getting user input list
usr_list = []
while(True):
    usr_in = input("Enter items of your list, type DONE to exit: ")
    if(usr_in.lower() == "done"): break;
    usr_list.append(usr_in)
in_str = " ".join(usr_list)

prompt = f'''
TASK:
Convert input into category totals.

RULES:
- Output ONLY valid JSON
- No explanation
- No extra text
- Keys must be category names (strings)
- Values must be integers
- Sum values of same category

EXAMPLE:
Input: swiggy 120 coffee 100 uber 300
Output:
{{"food": 220, "transport": 300}}

INPUT:
{in_str}

OUTPUT:
'''


import json

MAX_RETRIES = 7
success = False

for attempt in range(MAX_RETRIES):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": prompt,
                "stream": False
            },
            timeout=5
        )

        response.raise_for_status()

        data = response.json()
        raw_output = data["response"].strip()

        # extract JSON part
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1

        if start == -1 or end == -1:
            raise InvalidOutputError("No JSON found")

        json_str = raw_output[start:end]
        parsed = json.loads(json_str)

        # validate structure
        for key, value in parsed.items():
            if not isinstance(key, str):
                raise InvalidOutputError("Invalid key type")
            if not isinstance(value, int):
                raise InvalidOutputError("Invalid value type")

        # success
        success = True
        break

    except (requests.exceptions.RequestException,
            json.JSONDecodeError,
            KeyError,
            InvalidOutputError) as e:

        print(f"Attempt {attempt+1} failed:", e)

# after loop
if not success:
    print("Failed after all retries")
    exit()

print(parsed)