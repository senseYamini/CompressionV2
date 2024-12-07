import socket
import csv
import numpy as np
from datetime import datetime

tcp_ip = "" #self ip
tcp_port = 5005

#create the tcp socket
server_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_soc.bind((tcp_ip, tcp_port))
server_soc.listen(1)

print(f"server is listening on {tcp_ip}:{tcp_port}")
conn, addr = server_soc.accept()

#get ip address of the client
client_ip = addr[0]
print(f'client ip address:{client_ip}')

#open a csv file to write the recieved data
output_csv_file = "received_data_with_ip.csv"
csv_file = open(output_csv_file, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write CSV header
csv_writer.writerow(['Server_Timestamp', 'Client_IP', 
                     'Original_Timestamp', 'Data...'])


# For storing data in a NumPy array
received_data_list = []

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

         # Decode the received data to a CSV string
        csv_string = data.decode('utf-8')

        # Parse the CSV string back into rows
        rows = [list(map(float, row.split(','))) for row in csv_string.split('\n') if row.strip()]

        # Add server's timestamp and client IP to each row and write to CSV
        server_timestamp = datetime.now().timestamp()
        for row in rows:
            csv_writer.writerow([server_timestamp, client_ip] + row)

        # Add to NumPy data list
        for row in rows:
            received_data_list.append([server_timestamp, client_ip] + row)

        print(f"Received and logged {len(rows)} rows.")

finally:
    # Close the file and socket
    csv_file.close()
    conn.close()
    server_soc.close()

    # Convert the data to a NumPy array and save it
    np_array = np.array(received_data_list, dtype=object)  # Use dtype=object to store IP addresses
    np.save("received_data_with_ip.npy", np_array)
    print(f"Saved received data to {output_csv_file} and received_data_with_ip.npy.")
