import numpy as np
import os
import socket
import time

# Parameters for the distribution of rowcounts
mean_rowcount = 1049  # Mean of the rowcounts (samples per second)
std_rowcount = np.sqrt(42166)  # Standard deviation of the rowcounts

# Size of each sample in bytes
sample_size = 448  # bytes

# Total number of seconds
total_seconds = 60  # You can modify this as needed

# UDP socket setup
server_address = ('127.0.0.1', 5005)  # Server IP and port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Generate data dynamically
for second in range(1, total_seconds + 1):
    # Dynamically generate the number of samples for the current second
    sample_count = int(np.random.normal(mean_rowcount, std_rowcount))
    
    # Ensure the sample count is non-negative
    sample_count = max(sample_count, 0)
    
    # Generate random data for the current second as a NumPy array
    data = np.random.randint(0, 256, size=(sample_count, sample_size), dtype=np.uint8)  # Random bytes in range 0-255
    
    # Serialize data as bytes
    data_bytes = data.tobytes()
    
    # Send data to the server
    sock.sendto(data_bytes, server_address)
    
    # Print status
    print(f"Sent {sample_count} samples ({sample_count * sample_size} bytes) for second {second}.")
    
    # Sleep to simulate real-time data generation (1 second interval)
    time.sleep(1)

print("Data transmission complete.")
sock.close()
