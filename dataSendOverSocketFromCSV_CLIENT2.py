import pandas as pd
import numpy as np
import time
import socket

# Server details
TCP_IP = "10.21.232.163"  # Replace with your server's IP address
TCP_PORT = 5005           # Replace with your server's port

# Load the CSV file
csv_file = "3_timedomain.csv"  # Replace with your actual CSV file path
data = pd.read_csv(csv_file, header=None)  # Load CSV without headers (optional)

# Sort data by the first column (time) to ensure correct timing order
data = data.sort_values(by=0)  # Assumes first column is at index 0

# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))
print(f"Connected to server at {TCP_IP}:{TCP_PORT}")

# Get the simulation start time
simulation_start_time = time.time()

try:
    # Iterate through each row
    for _, row in data.iterrows():
        # Extract the target time and the rest of the row
        target_time = row[0]  # Time from the first column
        row_data = row[1:].to_numpy()  # Convert the rest of the row to a NumPy array

        # Calculate the time to wait until sending this row
        current_time = time.time()
        elapsed_time = current_time - simulation_start_time
        wait_time = target_time - elapsed_time

        # Wait if the target time is in the future
        if wait_time > 0:
            time.sleep(wait_time)

        # Convert the row data to bytes
        row_bytes = ",".join(map(str, row_data)).encode('utf-8')

        # Send the length of the row first
        sock.sendall(len(row_bytes).to_bytes(4, 'big'))

        # Send the actual row data
        sock.sendall(row_bytes)

        # Log the sent data
        print(f"At time {time.time() - simulation_start_time:.2f}s, sent row: {row_data}")

finally:
    # Close the socket after sending all data
    sock.close()
    print("All rows sent and connection closed.")
