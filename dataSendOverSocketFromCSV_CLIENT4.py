import pandas as pd
import numpy as np
import time
import socket

'''
continous wait
'''

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

# Start time reference
simulation_start_time = time.perf_counter()

try:
    row_index = 0  # Track the index of the current row

    while row_index < len(data):
        # Get the target time for the current row
        target_time = data.iloc[row_index, 0]  # Time from the first column

        # Get the current elapsed time since simulation start
        elapsed_time = time.perf_counter() - simulation_start_time

        # Check if it's time to send the row
        if elapsed_time >= target_time:
            # Get the row data and convert it to a NumPy array
            row_data = data.iloc[row_index, 1:].to_numpy()  # Exclude the time column

            # Convert the row data to a string and add a newline
            row_string = ",".join(map(str, row_data)) + "\n"

            # Send the row string
            sock.sendall(row_string.encode('utf-8'))

            # Log the sent data
            print(f"At time {elapsed_time:.2f}s, sent row: {row_data}")

            # Move to the next row
            row_index += 1
        else:
            # If it's not time yet, perform a non-blocking check
            time.sleep(0.001)  # Prevent busy-waiting by pausing briefly

finally:
    # Close the socket after sending all data
    sock.close()
    print("All rows sent and connection closed.")
