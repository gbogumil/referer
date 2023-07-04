import socket
import json
import struct
import time
import random

# The IP and port of the server
SERVER_IP = "localhost"
SERVER_PORT = 4776

# Connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))

# A list of observations
observations = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]]

def printbytearray(bytes):
    return [b for b in bytes]

def observation_generator():
    for _ in range(10):
        yield [random.random() for _ in range(10)]

sendcounter = 0

for observation in observation_generator():
    print(f'---=== sent {sendcounter} observations')
    # Prepare the data
    data = b"\x00" + b"".join(struct.pack('<f', f) for f in observation)
    print(f'data prepared {data}')
    # Send the length of data
    datalength = len(data)
    datalengthbytes = struct.pack('<i', datalength)
    print(f'len and bytes {datalength} - [{printbytearray(datalengthbytes)}]')
    client_socket.sendall(datalengthbytes + data)
    sendcounter += 1
    print('sent len + data')
    time.sleep(0.2)
    print('awake')
    # Wait for response
    bytes = client_socket.recv(4)
    response_length = struct.unpack('<i', bytes)[0]
    print(f'bytes received({len(bytes)}) {printbytearray(bytes)} {response_length}')
    response = b""
    while len(response) < response_length:
        print(f'looping response buffer {len(response)} is less than {response_length}')
        response += client_socket.recv(response_length - len(response))
    print(f"Received response: {printbytearray(response)}")
    floats = struct.unpack('<f', response)
    print(f'------- {floats} -------')

client_socket.close()

