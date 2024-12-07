import socket
import csv
import numpy as np
from datetime import datetime

tcp_ip = "10.21.232.163"  # Server IP
tcp_port = 5005           # Server Port

# Create the TCP socket
server_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_soc.bind((tcp_ip, tcp_port))
server_soc.listen(1)

print(f"Server is listening on {tcp_ip}:{tcp_port}")
conn, addr = server_soc.accept()

# Get IP address of the client
client_ip = addr[0]
print(f"Client IP address: {client_ip}")

# Open a CSV file to write the received data
output_csv_file = "received_data_with_ip.csv"
csv_file = open(output_csv_file, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write CSV header
csv_writer.writerow(['Server_Timestamp', 'Client_IP', 'Original_Timestamp', 'Data...'])

# For storing data in a NumPy array
received_data_list = []

try:
    buffer = ""  # Buffer for partial data
    while True:
        # Receive data in chunks
        data = conn.recv(4096).decode('utf-8')
        if not data:
            break

        buffer += data  # Append new data to the buffer

        # Process complete rows
        while "\n" in buffer:
            row, buffer = buffer.split("\n", 1)  # Extract one row
            if row.strip():  # Skip empty rows
                # Parse the row into a list of floats
                try:
                    parsed_row = [float(value) for value in row.split(",")]
                    server_timestamp = datetime.now().timestamp()
                    csv_writer.writerow([server_timestamp, client_ip] + parsed_row)
                    received_data_list.append([server_timestamp, client_ip] + parsed_row)
                    print(f"Received and logged row: {parsed_row}")
                except ValueError as e:
                    print(f"Error parsing row: {row}. Error: {e}")

finally:
    # Close the file and socket
    csv_file.close()
    conn.close()
    server_soc.close()

    # Convert the data to a NumPy array and save it
    np_array = np.array(received_data_list, dtype=object)  
    np.save("received_data_with_ip.npy", np_array)
    print(f"Saved received data to {output_csv_file} and received_data_with_ip.npy.")
