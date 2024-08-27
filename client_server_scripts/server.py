import os
import socket
import threading
from groq import Groq
from datetime import datetime
import json

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
print(f"Using Groq API Key: {api_key}")
client = Groq(api_key=api_key)

def get_response_from_groq(prompt):
    print("Entering get_response_from_groq function")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )
        print("Groq API request successful")
        response_content = chat_completion.choices[0].message.content
        print(f"Groq response: {response_content}")
        return response_content
    except Exception as e:
        print(f"Error getting response from Groq: {e}")
        return "Error"

def handle_client(client_socket):
    print("Handling client")
    try:
        # Initialize an empty buffer
        buffer = b''
        
        while True:
            # Receive data in chunks
            chunk = client_socket.recv(1024)
            if not chunk:
                break  # Exit loop if no more data is received
            
            buffer += chunk  # Append received data to buffer
            
            # Try to decode the buffer; handle it as a complete response
            try:
                prompt = buffer.decode()
                print(f"Received prompt: {prompt}")

                # Handle the prompt and get the response
                response_content = get_response_from_groq(prompt)

                # Prepare and send the response
                response_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': prompt,
                    'response': response_content,
                }
                response_json = json.dumps(response_data)
                print(f"Sending response: {response_json}")
                
                client_socket.sendall(response_json.encode())
                break  # Exit the loop after processing a complete request
            except UnicodeDecodeError:
                # If decoding fails, continue to accumulate data
                print("Incomplete data received, waiting for more.")
                continue

    except Exception as e:
        print(f"Error while handling client: {e}")
    finally:
        client_socket.close()

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 12345))
    server_socket.listen(5)
    print("Server listening on port 12345...")
    
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == "__main__":
    start_server()
