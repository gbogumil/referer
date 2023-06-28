import base64
import socket
import json
import numpy as np
from GapLearner import GapLearner
import traceback
import logging

class DataPacket:
    def __init__(self, method, data):
        self.method = method
        self.data = base64.b64encode(data).decode()

class ResponsePacket:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self.data = base64.b64encode(data).decode()

class ByteQueue:
    def __init__(self):
        self.buffer = bytearray()

    def receive(self, bytes):
        self.buffer.extend(bytes)
    
    def get(self, length):
        result = self.buffer[:length]
        self.buffer = self.buffer[length:]
        return result
    
    def __len__(self):
        return len(self.buffer)

class GapServer:
    def __init__(self, host='localhost', port=4776):
        self.logger = logging.getLogger('gaplogger')
        self.logger.info('initializing')
        self.learner = GapLearner(observation_size=10)
        self.logger.info('created GapLearner')
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.settimeout(1.0)
        self.server.setblocking(True)
        self.buffer = ByteQueue()
        self.action_size = 1
        self.action_infer = 0
        self.action_stats = 1
        self.action_sizes = [40, # infer size
                             0] # stats size
        self.logger.info('binding')
        self.server.bind((host, port))
        self.logger.info('listening')
        self.server.listen()
        self.logger.info('initialized')
        self.statcountreset = 100
        self.statcounter = self.statcountreset

    def start(self):
        host, port = self.server.getsockname()
        self.logger.info(f"Server is waiting for connection on {host}:{port}")
        while True:
            conn, addr = self.server.accept()  # accept a connection
            self.logger.info(f"Client connected from {addr}")
            self.handle_client(conn)

    def handle_client(self, conn):
        while True:
            try:
                recv = conn.recv(4) # message length bytes
                message_length = int.from_bytes(recv, byteorder='little')
                recv = conn.recv(message_length)
                self.buffer.receive(recv)
                while self.can_process_action():
                    action = int.from_bytes(self.buffer.get(self.action_size), byteorder='little')
                    self.logger.debug(f'processing action {action}')
                    if action == self.action_infer:
                        observation = self.buffer.get(self.action_sizes[self.action_infer])
                        self.logger.debug(f'observed {self.bytestostring(observation)}')
                        result = self.infer(observation)
                        self.logger.debug(f'result {self.bytestostring(result)}')
                    elif action == self.action_stats:
                        result = self.stats()
                    else:
                        result = 'Error: Invalid method'
                    if len(result):
                        conn.sendall(result)  # send result back over the connection
                        self.logger.debug('sent')

            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                break

        conn.close()  # close the connection when the client disconnects or an error occurs
        logger.info("Connection closed")

    def can_process_action(self):
        if len(self.buffer) >= self.action_size:
            action = int.from_bytes(self.buffer.buffer[:self.action_size], byteorder='little')
            self.logger.debug(f'action {action}')
            return len(self.buffer) >= self.action_size + self.action_sizes[action]
        return False

    def infer(self, data):
        self.statcounter -= 1
        if self.statcounter <= 0:
            self.statcounter = self.statcountreset
            _ = self.stats()
        elif self.statcounter % 10 == 0:
            self.logger.info(self.statcounter)
        # Here, data is now a list of float values
        self.logger.debug(f'gapserver: {data}')
        inference = self.learner.predict(data, None, True)
        self.logger.debug(f'gapserver: {inference}')
        if inference is None:
            inference = data
        else: 
            inference = inference.tobytes()
        return inference

    def stats(self):
        # Return ML model statistics here as bytes and a status code
        stats = self.learner.stats()
        self.logger.info(f'stats\n{stats}')
        return None
    
    def bytestostring(self, bytes):
        formatted_data = '-'.join(f'{byte:02X}' for byte in bytes)
        return formatted_data


if __name__ == '__main__':

    # Create a logger
    logger = logging.getLogger('gaplogger')
    logger.setLevel(logging.DEBUG)  # Set the minimum log level of messages this logger will handle

    # Create a file handler
    file_handler = logging.FileHandler('gapserver.log')
    file_handler.setLevel(logging.INFO)  # Set the minimum log level of messages this handler will handle

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Now you can log to both the file and the console via the logger
    logger.info('gaplogger configured')
    GapServer().start()
    logger.info('gapserver started')

