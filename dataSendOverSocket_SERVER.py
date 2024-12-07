import socket
import csv
import time
from datetime import datetime

#udp socket setup
server_address = ('127.0.0.1', 5005)#(ip.port)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(server_address)

csv_file = 'received_data.csv'
with open(csv_file, mode = 'w', newline = '')as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Data'])#header row

print("Server is listening for data")

try:
    while True:
        data, addr = sock.recvfrom(65534)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        data_size = len(data)

        with open(csv_file, model='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, data])
        
        print(f'Received {data_size} bytes from {addr} at {timestamp}')

except KeyboardInterrupt:
    print("server shutting down")

sock.close()



