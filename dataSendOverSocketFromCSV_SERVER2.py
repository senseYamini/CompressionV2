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

def sanitize_row(row):
    """Sanitize a row string by removing invalid characters."""
    try:
        # Replace invalid characters and split into valid float values
        sanitized_values = [float(value.strip()) for value in row.split(',') if value.strip()]
        return sanitized_values
    except ValueError as e:
        print(f"Error sanitizing row: {row}. Error: {e}")
        return None  # Skip invalid rows

try:
    while True:
        # Receive the length of the incoming data
        data_length = conn.recv(4)
        if not data_length:
            break

        data_length = int.from_bytes(data_length, 'big')

        # Receive the actual data
        data = b""
        while len(data) < data_length:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        # Debugging: Print raw data length and a sample
        print(f"Received raw data length: {len(data)}")
        print(f"Raw data sample: {data[:50]}...")  # Truncated for display

        # Decode the received data to a CSV string
        try:
            csv_string = data.decode('utf-8')
        except UnicodeDecodeError as e:
            print(f"Decoding error: {e}. Replacing invalid characters.")
            csv_string = data.decode('utf-8', errors='replace')

        # Parse and sanitize rows
        sanitized_rows = []
        for row in csv_string.split('\n'):
            if row.strip():  # Skip empty rows
                sanitized_row = sanitize_row(row)
                if sanitized_row is not None:
                    sanitized_rows.append(sanitized_row)

        # Add server's timestamp and client IP to each row and write to CSV
        server_timestamp = datetime.now().timestamp()
        for row in sanitized_rows:
            csv_writer.writerow([server_timestamp, client_ip] + row)

        # Add to NumPy data list
        for row in sanitized_rows:
            received_data_list.append([server_timestamp, client_ip] + row)

        print(f"Received and logged {len(sanitized_rows)} rows.")

finally:
    # Close the file and socket
    csv_file.close()
    conn.close()
    server_soc.close()

    # Convert the data to a NumPy array and save it
    np_array = np.array(received_data_list, dtype=object)  
    np.save("received_data_with_ip.npy", np_array)
    print(f"Saved received data to {output_csv_file} and received_data_with_ip.npy.")
