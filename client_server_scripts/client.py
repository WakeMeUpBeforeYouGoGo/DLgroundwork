import socket
import json
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python client.py <output_file>")
        sys.exit(1)

    input_file = 'input.txt'
    output_file = sys.argv[1]

    server_address = ('localhost', 12345) #The server address that the client should use to connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Initializing the socket in order to connect
    client_socket.connect(server_address) #Connect at the given address
    print(f"Connected to server at {server_address}")

    with open(input_file, 'r') as file:
        prompts = file.readlines()

    responses = []

    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue

        client_socket.sendall(prompt.encode()) #Send everything and encode it in the byte format
        print(f"Sent prompt: {prompt}") #Sent promp to the sevrer, debugging

        response = b""
        while True:
            part = client_socket.recv(1024) #Recieving the response using .recv function in the form of bytes
            if not part:
                break
            response += part #Adding the response, this is done so thart the entire response is being taken and not just partial response

        response_str = response.decode() #Decoding from byte to string
        print(f"Received raw response: {response_str}")#Debugging 

        try:
            response_data = json.loads(response_str)
            print(f"Parsed response data: {response_data}")
        except json.JSONDecodeError as e:
            print(f"Error parsing response JSON: {e}")
            continue

        responses.append({
            'prompt': prompt,
            'response': response_data.get('response', ''),
        })

    with open(output_file, 'w') as file:
        json.dump(responses, file, indent=4)
    print(f"Responses saved to {output_file}") #Dump it in the json format into the output file

    client_socket.close() #End the connection 

if __name__ == "__main__":
    main()
