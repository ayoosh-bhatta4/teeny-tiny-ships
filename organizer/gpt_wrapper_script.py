import requests
usr_list = []
while(True):
    usr_in = input("Enter items of your list, type DONE to exit: ")
    if(usr_in.lower() == "done"): break;
    usr_list.append(usr_in)
in_str = " ".join(usr_list)

prompt = f'''INSTRUCTION: You are a system that will be given a list of items followed by their price. currency is rupees if not mentioned. your job is to replace the specific names of the items by their more generic category. For example, Uber beccomes transport. Swiggy becomes food and so on. If there are things which are in the same category, add up their prices. Only output the result. No explanantion. Output EXACTLY in this format: <category> <number> no extra words. 
            EXAMPLE: if input is "swiggy 120 coffee 100 uber 300", you output
                food 220
                transport 300
            INPUT: {in_str}
            OUTPUT:'''


try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json = {
            "model" : "phi",
            "prompt": prompt,
            "stream": False
        }
    )
except requests.exceptions.ConnectionError:
    print("Ollama not working")
except requests.exceptions.ConnectTimeout:
    print("Ollama connection timeout")
except requests.exceptions.InvalidURL:
    print("Wrong url")
except Exception as e:
    print("Unknown error when getting response from ollama")

data = response.json()
output = data["response"]

print(output)