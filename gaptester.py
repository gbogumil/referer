import GapLearner
import random
import logging
import Memory
import numpy as np
from flask import Flask, jsonify, render_template, request
import threading
import time
import json
import os
import glob

sendcounter = 0
# sample data params
observation_qty = 300
observation_length = 10
frame_size = 15
# learning input params
training_counter = 10
training_counter_reset = training_counter
evolved = False


delay = 0.001
training_count = 0
latest_observation = None
latest_inference = None

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



def dump(name, data):
    stringdata = json.dumps(data, indent=2)
    logger.info(f'name: {stringdata}')

    with open(f'{name}.json', 'w') as f:
        json.dump(data, f, indent=4)

def deleteevolutionstats():
    patterns = [f'{name} *.json' for name in names]
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            os.remove(file)

def statictests():
    deleteevolutionstats()

    for observation in observation_generator(observation_qty, observation_length, frame_size):
        time.sleep(delay)
        logger.info(f'---=== sent {sendcounter} observations')
        logger.info(f' sending {observation}')
        # Prepare the data

        if len(group.memories.memory) % training_counter_reset == 0:
            training_count += 1
            dump(f'stats before prune {training_count}', group.stats())
            group.prune()
            dump(f'stats after prune {training_count}', group.stats())
            group.train()
            dump(f'stats after training {training_count}', group.stats())

            group.evolve()
            dump(f'stats after evolve {training_count}', group.stats())

            training_counter = training_counter_reset
        else:
            training_counter -= 1
            

        inference = group.predict(observation, None)
        logger.info(f'inferred {inference}')
        latest_observation = observation.observation
        latest_inference = inference
    logger.info(f'stats in tester: {group.stats()}')

def shouldprune():
    # prune if there are more memories in the memory than in the bins
    bintotal = sum(len(bin) for bin in group.memories.memory_bins)
    if len(group.memories.memory) - bintotal > bintotal:
        return True
    return False

def shouldtrain():
    return training_counter <= 0

def shouldevolve():
    return not evolved and training_count % 3 == 1

def waitforserver():
    while True:
        if shouldprune():
            group.prune()
        if shouldtrain():
            training_counter = training_counter_reset
            training_count += 1
            evolved = False
            group.train()
        if shouldevolve():
            group.evolve()
            evolved = True

        time.sleep(0.025)

########
# FLASK config
app = Flask(__name__)
# app.config['DEBUG'] = True

def run_flask_app():
    training_counter = training_counter_reset
    host = 'localhost'
    port = 4776

    app.run(host=host, port=port)
    logger.info(f'FLASK: flask server running on {host}:{port}')

@app.route('/')
def index():
    ret = render_template('index.html')
    return ret

@app.route('/get_stats', methods=['GET'])
def get_stats():
    stats = group.stats()
    return jsonify(stats)

@app.route('/infer', methods=['POST'])
def get_infer():
    data = request.get_json()
    logger.debug(f'infer data {data}')
    observation = data['observation']
    observation = Memory.Observation(observation)
    inference = group.predict(observation)
    return jsonify({'inference': inference.tolist() if isinstance(inference, np.ndarray) else inference})

@app.route('/get_latestinference', methods=['GET'])
def get_latestinference():
    logger.debug('FLASK: get_latestinference')
    return {
        'observation': ', '.join([str(o) for o in latest_observation]) if latest_observation is not None else '',
        'inference': ', '.join([str(o) for o in latest_inference]) if latest_inference is not None else ''
        }

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

if __name__ == '__main__':

    names = ['stats before prune', 'stats after prune', 'stats after training', 'stats after evolve']

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

    group = GapLearner.GapGroup()

    waitforserver()
