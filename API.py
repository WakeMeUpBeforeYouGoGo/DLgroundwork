import os #interactwing with OS to get the API key, helps keeps the API safe within the system :)
import json # obvious reasons
from datetime import datetime # libray i initialized to record time
from groq import Groq #LLM model i am taking the output fromn

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def get_response_from_groq(prompt):
    # Send API request to Groq
    chat_completion = client.chat.completions.create( #invoking the groq client to do some stuff. specificlaly the one which deals with chat within the client module, and the completions within the chat module and so on on
        messages=[
            {
                "role": "user",  #coming from user
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",  #specifies the model, can remove this line but would lead to weird or default models ig,not used anywhere else
    )
    
    # Extract and return the response content
    return chat_completion.choices[0].message.content #sends the content to geenrate the result, also getting the first choice annswer with the index[0]

def main():
    # Read prompts from a text file
    input_file = 'input.txt'
    output_file = 'responses.json'

    # Read all lines from the input file
    with open(input_file, 'r') as file:
        prompts = file.readlines() 

    # Prepare data to save to JSON
    responses = []
    
    for prompt in prompts:
        prompt = prompt.strip()  # Clean any extra whitespace
        if prompt:
            response_content = get_response_from_groq(prompt)
            response_data = {
                'timestamp': datetime.utcnow().isoformat(), #timestuff
                'source': prompt,
                'response': response_content,
            }
            responses.append(response_data)

    # Save responses to a JSON file
    with open(output_file, 'w') as file:
        json.dump(responses, file, indent=4)

if __name__ == "__main__":
    main()


