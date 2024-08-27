import socket
import json
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python client.py <output_file>")
        sys.exit(1)

    input_file = 'input.txt'
    output_file = sys.argv[1]

    server_address = ('localhost', 12345)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)
    print(f"Connected to server at {server_address}")

    with open(input_file, 'r') as file:
        prompts = file.readlines()

    responses = []

    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue

        client_socket.sendall(prompt.encode())
        print(f"Sent prompt: {prompt}")

        response = b""
        while True:
            part = client_socket.recv(1024)
            if not part:
                break
            response += part

        response_str = response.decode()
        print(f"Received raw response: {response_str}")

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
    print(f"Responses saved to {output_file}")

    client_socket.close()

if __name__ == "__main__":
    main()
