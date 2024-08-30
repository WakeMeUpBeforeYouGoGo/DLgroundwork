import os
import socket
import threading
from groq import Groq
from datetime import datetime
import json

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY") #Intializing the API, you need to get yours own :)
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
print(f"Using Groq API Key: {api_key}")
client = Groq(api_key=api_key)#Exception function to check if the API is working or not

def get_response_from_groq(prompt):
    print("Entering get_response_from_groq function")
    try:
        chat_completion = client.chat.completions.create( #Client module, and the chat module, which completes the answer, and create module
            messages=[
                {"role": "user", "content": prompt} #Role is defined to be user, and content is being used to whatever prompt i am sending it
            ],
            model="llama3-8b-8192", #Model of the GROQ API
        )
        print("Groq API request successful") #Debugging 
        response_content = chat_completion.choices[0].message.content #Getting the first response out of all responses there is 
        print(f"Groq response: {response_content}") #Debugging statement again, not needed, can be deleted
        return response_content 
    except Exception as e:
        print(f"Error getting response from Groq: {e}")
        return "Error"

def handle_client(client_socket):
    print("Handling client")
    try:
        # Initialize an empty buffer
        buffer = b'' #Initializing an empty buffer to store the the query, whatever it is
        
        while True:
            # Receive data in chunks
            chunk = client_socket.recv(1024) #The query is recieved, and keeps recieving the string prompt, in the form of byte string with .recv. Max of 1024 bytes
            if not chunk:
                break  # Exit loop if no more data is received
            
            buffer += chunk  # Append received data to buffer, basically to avoid half the prompt from not getting stored 
            
            # Try to decode the buffer; handle it as a complete response
            try:
                prompt = buffer.decode() #Decode it from bytes to string litera type
                print(f"Received prompt: {prompt}") #Debugging 

                # Handle the prompt and get the response
                response_content = get_response_from_groq(prompt) #Get the reponse 

                # Prepare and send the response
                response_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': prompt,
                    'response': response_content,
                }
                response_json = json.dumps(response_data) #Dump the response into a json format
                print(f"Sending response: {response_json}") #Debugging
                
                client_socket.sendall(response_json.encode())
                break  # Exit the loop after processing a complete request
            except UnicodeDecodeError:
                # If decoding fails, continue to accumulate data
                print("Incomplete data received, waiting for more.")
                continue

    except Exception as e:
        print(f"Error while handling client: {e}")
    finally:
        client_socket.close() #End connection 

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Initializing the server with the socket stream
    server_socket.bind(('0.0.0.0', 12345)) #Initializing the server with the IP and the port 
    server_socket.listen(5) #Can take 5 of them at one time and remaining would have to wait in queue 
    print("Server listening on port 12345...") #Debugging
    
    while True:
        client_socket, addr = server_socket.accept() 
        print(f"Accepted connection from {addr}")
        
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == "__main__":
    start_server()
