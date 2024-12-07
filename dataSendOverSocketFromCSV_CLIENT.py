'''
Using TCP ensures reliable transmission, 
ordered delivery, and no need for manual 
chunking because TCP handles packet 
segmentation and reassembly.
'''

import pandas as pd
import numpy as np
import socket
from datetime import datetime
import time
import pickle

#server details
tcp_ip = "192.168.1.109" #put server ip
tcp_port = 5005 #replace server port number

#load the csv file
csv_file = "3_updated.csv"
data = pd.read_csv(csv_file)

#create a tcp socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((tcp_ip, tcp_port))

for second in sorted(data.iloc[:,0].unique()):
    #get all rows with the first column value 
    rows_for_second = data[data.iloc[:, 0] == second]

    #convert the row to numpy array
    rows_as_array = rows_for_second.to_numpy()

    #get system timestamp
    system_timestamp = datetime.now().timestamp()

    #add timestamp as column to the data
    timestamp_col = np.full((rows_as_array.shape[0],1), system_timestamp)
    data_with_timestamp = np.hstack((timestamp_col, rows_as_array))

    #convert to plain-text csv format
    csv_string = "\n".join([",".join(map(str, row)) 
                            for row in data_with_timestamp])

    #encode the csv string to bytes
    data_bytes = csv_string.encode('utf-8')

    #send data len
    sock.sendall(len(data_bytes).to_bytes(4,'big'))

    #send actual data
    sock.sendall(data_bytes)

    #print status
    print(f"Sent data for second {second} with timestamp {system_timestamp}.")
    
    # Wait for 1 second
    time.sleep(1)

print("Data transmission complete.")
sock.close()

