import GapLearner
import random
import logging
import Memory
import numpy as np
from flask import Flask, jsonify
import threading
import time

def printbytearray(bytes):
    return [b for b in bytes]

def observation_generator():
    observation_qty = 1000
    observation_length = 10
    frame_size = 8
    # random sequence
    # sequence = [[random.random() for _ in range(observation_length)] for _ in range(frame_size)]
    # increasing sequence
    randomobservation = np.array([random.random() for _ in range(observation_length)])
    framesequence = [e*(1+i)*1.1 for i, e 
                     in enumerate([randomobservation 
                                   for _ in range(frame_size)])]

    for i in range(observation_qty):
        yield Memory.Observation(framesequence[i % frame_size])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_status', methods=['GET'])
def get_stats():
    stats = group.stats()
    return jsonify(stats)

def run_flask_app():
    port = 5000
    app.run(port)
    logger.info(f'flask server running on port {port}')

if __name__ == '__main__':

    # Create a logger
    logger = logging.getLogger('gaplogger')
    logger.setLevel(logging.DEBUG)  # Set the minimum log level of messages this logger will handle
    #logger.setLevel(logging.INFO)  # Set the minimum log level of messages this logger will handle

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

    t = threading.Thread(target=run_flask_app)
    t.daemon = True
    t.start()
    logger.info('flask server thread started')

    sendcounter = 0
    group = GapLearner.GapGroup()
    
    
    training_counter = 100
    training_counter_reset = training_counter

    for observation in observation_generator():
        time.sleep(0.001)
        logger.info(f'---=== sent {sendcounter} observations')
        logger.info(f' sending {observation}')
        # Prepare the data

        if training_counter <= 0:
            group.prune()
            group.train()
            training_counter = training_counter_reset
            logger.info('training complete')
            logger.info(f'stats:\n{group.stats()}')
        else:
            training_counter -= 1
            logger.info(f'training_counter {training_counter}')
            

        inference = group.predict(observation, None)
        logger.info(f'inferred {inference}')
    logger.info(f'stats in tester: {group.stats()}')
