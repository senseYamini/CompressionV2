'''
server code to handle multiple clients at once
using multithreading.
'''


import socket
import csv
import numpy as np
from datetime import datetime
import threading

tcp_ip = "10.21.232.163"  # Server IP
tcp_port = 5005           # Server Port

# Create the TCP socket
server_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_soc.bind((tcp_ip, tcp_port))
server_soc.listen(5)  # Listen for up to 5 simultaneous connections
print(f"Server is listening on {tcp_ip}:{tcp_port}")

# Lock to prevent race conditions when writing to shared resources
lock = threading.Lock()

# Open a CSV file to write the received data (shared by all threads)
output_csv_file = "received_data_with_ip.csv"
csv_file = open(output_csv_file, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write CSV header
csv_writer.writerow(['Server_Timestamp', 'Client_IP', 'Original_Timestamp', 'Data...'])

# For storing data in a NumPy array (shared by all threads)
received_data_list = []


def handle_client(conn, addr):
    """Handle communication with a single client."""
    client_ip = addr[0]
    print(f"Connected to client: {client_ip}")

    buffer = ""  # Buffer for partial data
    try:
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

                        # Write to shared resources with a lock
                        with lock:
                            csv_writer.writerow([server_timestamp, client_ip] + parsed_row)
                            received_data_list.append([server_timestamp, client_ip] + parsed_row)

                        print(f"Received and logged row from {client_ip}: {parsed_row}")
                    except ValueError as e:
                        print(f"Error parsing row from {client_ip}: {row}. Error: {e}")
    except Exception as e:
        print(f"Error with client {client_ip}: {e}")
    finally:
        print(f"Connection closed with client: {client_ip}")
        conn.close()


try:
    while True:
        conn, addr = server_soc.accept()
        # Start a new thread for each client
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.daemon = True  # Ensures threads close when the main program exits
        client_thread.start()

finally:
    # Close the file and socket when shutting down
    csv_file.close()
    server_soc.close()

    # Convert the data to a NumPy array and save it
    np_array = np.array(received_data_list, dtype=object)  
    np.save("received_data_with_ip.npy", np_array)
    print(f"Saved received data to {output_csv_file} and received_data_with_ip.npy.")
