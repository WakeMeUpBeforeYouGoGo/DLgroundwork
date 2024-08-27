#!/bin/bash

# Start the server in the background
echo "Starting server..."
python3 server.py &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

# Wait for the server to start
sleep 2

# Start clients with different output files using a loop
for i in {1..3}
do
    OUTPUT_FILE="output${i}.json"
    echo "Starting client $i with output file $OUTPUT_FILE..."
    python3 client.py $OUTPUT_FILE &
done

# Wait for all clients to complete
wait

# Stop the server
echo "Stopping server..."
kill $SERVER_PID
echo "Server stopped."
