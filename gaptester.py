import GapLearner
import random
import logging
import Memory
import numpy as np
from flask import Flask, jsonify, render_template
import threading
import time
import json
import os
import glob

def printbytearray(bytes):
    return [b for b in bytes]

def observation_generator(observation_qty, observation_length, frame_size):
    # random sequence
    # sequence = [[random.random() for _ in range(observation_length)] for _ in range(frame_size)]
    # increasing sequence
    randomobservation = np.array([random.random() for _ in range(observation_length)])
    framesequence = [e*(1+i)*1.1 for i, e 
                     in enumerate([randomobservation 
                                   for _ in range(frame_size)])]

    for i in range(observation_qty):
        yield Memory.Observation(framesequence[i % frame_size], i)

def loadJsonLog(name, instance):
    try:
        with open(f'{name} {instance}.json', 'r') as f:
            return json.load(f)
    except:
        return None



app = Flask(__name__)
# app.config['DEBUG'] = True

@app.route('/')
def index():
    print('FLASK: rendering index')
    ret = render_template('index.html')
    print('FLASK: rendered index')
    return ret

@app.route('/get_stats', methods=['GET'])
def get_stats():
    print('FLASK: getting stats')
    stats = group.stats()
    print('FLASK: got stats')
    return jsonify(stats)

@app.route('/infer', methods=['GET'])
def get_infer():
    print('FLASK: infer')

names = ['stats before prune', 'stats after prune', 'stats after training', 'stats after evolve']

def deleteevolutionstats():
    patterns = [f'{name} *.json' for name in names]
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            os.remove(file)

@app.route('/get_evolutionstats', methods=['GET'])
def get_evolutionstats():
    # loop through the local folder for stats before / after json files
    print('FLASK: getting evolution stats')
    
    ret = []
    instance = 0
    while instance < 20:
        instance += 1
        print(f'appending instance {instance}')
        instancedata = {name: loadJsonLog(name, instance) for name in names}
        if all(instancedata[name] is None for name in names):
            break
        ret.append(instancedata)
    print('FLASK: got evolution stats')
    return {f'instance {i}': data for i, data in enumerate(ret)}

def run_flask_app():
    host = 'localhost'
    port = 5000
    app.run(host=host, port=port)
    logger.info(f'FLASK: flask server running on {host}:{port}')

def dump(name, data):
    stringdata = json.dumps(data, indent=2)
    logger.info(f'name: {stringdata}')

    with open(f'{name}.json', 'w') as f:
        json.dump(data, f, indent=4)

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

    deleteevolutionstats()

    t = threading.Thread(target=run_flask_app)
    t.daemon = True
    t.start()
    logger.info('flask server thread started')

    sendcounter = 0
    group = GapLearner.GapGroup()
    
    
    # sample data params
    observation_qty = 3000
    observation_length = 10
    frame_size = 15
    # learning input params
    training_counter = 100
    training_counter_reset = training_counter
    delay = 0.001
    training_count = 0

    for observation in observation_generator(observation_qty, observation_length, frame_size):
        time.sleep(delay)
        logger.info(f'---=== sent {sendcounter} observations')
        logger.info(f' sending {observation}')
        # Prepare the data

        if training_counter <= 0:
            training_count += 1
            logger.info(f'stats before prune: {group.stats()}')
            dump(f'stats before prune {training_count}', group.stats())
            group.prune()
            logger.info(f'stats after prune: {group.stats()}')
            dump(f'stats after prune {training_count}', group.stats())
            group.train()
            logger.info(f'stats after training:\n{group.stats()}')
            dump(f'stats after training {training_count}', group.stats())

            group.evolve()
            logger.info(f'stats after evolve:\n{group.stats()}')
            dump(f'stats after evolve {training_count}', group.stats())

            training_counter = training_counter_reset
        else:
            training_counter -= 1
            logger.debug(f'training_counter {training_counter}')
            

        inference = group.predict(observation, None)
        logger.info(f'inferred {inference}')
    logger.info(f'stats in tester: {group.stats()}')

