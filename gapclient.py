import socket
import json
import struct
import time

# The IP and port of the server
SERVER_IP = "localhost"
SERVER_PORT = 4776

# Connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))

# A list of observations
observations = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]]

for observation in observations:
    # Prepare the data
    data = b"\x00" + b"".join(struct.pack('<f', f) for f in observation)
    # Send the length of data
    datalength = len(data)
    datalengthbytes = struct.pack('<i', datalength)
    client_socket.sendall(datalengthbytes)
    # Send the data
    client_socket.sendall(data)
    time.sleep(5)
    # Wait for response
    bytes = client_socket.recv(4)
    print(f'bytes received({len(bytes)}) {bytes}')
    response_length = struct.unpack('<i', client_socket.recv(4))[0]
    response = b""
    while len(response) < response_length:
        response += client_socket.recv(response_length - len(response))
    print(f"Received response: {response.decode('utf-8')}")

client_socket.close()
