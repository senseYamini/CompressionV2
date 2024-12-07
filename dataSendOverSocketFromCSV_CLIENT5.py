'''
infintely send data from the file while updating the timestamps
while sending.
'''

import pandas as pd
import numpy as np
import time
import socket

# Server details
TCP_IP = "10.21.232.163"  # Replace with your server's IP address
TCP_PORT = 5005           # Replace with your server's port

# Load the CSV file
csv_file = "3_timedomain.csv" 
data = pd.read_csv(csv_file, header=None, skiprows=1) 

data = data.sort_values(by=0)  

# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))
print(f"Connected to server at {TCP_IP}:{TCP_PORT}")

try:
    simulation_start_time = time.time()  # Start the simulation clock
    cumulative_time_offset = 0.0  # Initialize the cumulative time offset

    while True:  # Infinite loop to keep sending the file
        for _, row in data.iterrows():
            original_target_time = row[0] 
            row_data = row.to_numpy() 

            
            updated_target_time = original_target_time + cumulative_time_offset

            current_time = time.time()
            elapsed_time = current_time - simulation_start_time

            wait_time = updated_target_time - elapsed_time

            if wait_time > 0:
                time.sleep(wait_time)

            # Update the first column value with the updated target time
            row_data[0] = updated_target_time

            row_string = ",".join(map(str, row_data)) + "\n"

            sock.sendall(row_string.encode('utf-8'))

            # Log the sent data
            # print(f"At time {elapsed_time:.2f}s, sent row: {row_data}")

        # Update the cumulative time offset after finishing one file iteration
        cumulative_time_offset += data.iloc[-1, 0]  # Add the last row's time value to the offset
        print(f"File transmission completed. Cumulative offset updated to {cumulative_time_offset:.3f}")

finally:
    # Close the socket after the program is stopped
    sock.close()
    print("Connection closed.")
