'''
Created on Apr 24, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

Script to generate configurations for acrobot experiment.

'''

import json

import os
experimentDir = '../experiment/acrobot'
if not os.path.exists( experimentDir ):
    os.makedirs( experimentDir )

def create_ac_all():

    config = {}

    params = {}
    params['-e'] = [100]
    params['-i'] = [1500]
    params['-R'] = [20]
    params['-a'] = [.1]
    params['-b'] = [.005]
    params['-A'] = ['Q', 'GQ', 'GQ2', 'PGQ', 'PGQ2']
    params['--behaviorTemperature'] = [1.1]
    params['--targetTemperature'] = [.5]


    config['parameter'] = params
    config['name'] = 'ac_all'
    config['script'] = 'acrobot.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/acrobot_experiment_config_ac_all.json', 'wb' ) as fp:
        json.dump( config, fp )

def create_ac_all2():

    config = {}

    params = {}
    params['-e'] = [100]
    params['-i'] = [1500]
    params['-R'] = [20]
    params['-a'] = [.005, .01, .05, .1]
    params['-b'] = [.005, .01, .05, .1]
    params['-A'] = ['Q', 'GQ', 'GQ2', 'PGQ', 'PGQ2']
    params['--behaviorTemperature'] = [1.5]
    params['--targetTemperature'] = [.2]


    config['parameter'] = params
    config['name'] = 'ac_all'
    config['script'] = 'acrobot.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/acrobot_experiment_config_ac_all2.json', 'wb' ) as fp:
        json.dump( config, fp )

def create_ac_GQ2_PGQ2():

    config = {}

    params = {}
    params['-e'] = [100]
    params['-i'] = [1500]
    params['-R'] = [20]
    params['-a'] = [.001, .005, .01, .05, .1, .2]
    params['-b'] = [.001, .005, .01, .05, .1, .2]
    params['-A'] = ['GQ2', 'PGQ2']
    params['--behaviorTemperature'] = [1.1]
    params['--targetTemperature'] = [.5]


    config['parameter'] = params
    config['name'] = 'ac_GQ2_PGQ2'
    config['script'] = 'acrobot.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/acrobot_experiment_config_ac_GQ2_PGQ2.json', 'wb' ) as fp:
        json.dump( config, fp )

def create_ac_all_500():

    config = {}

    params = {}
    params['-e'] = [500]
    params['-i'] = [1500]
    params['-R'] = [1] * 20
    params['-a'] = [.1]
    params['-b'] = [.005]
    params['-A'] = ['Q', 'GQ', 'PGQ']
    params['--behaviorTemperature'] = [1.1]
    params['--targetTemperature'] = [.5]


    config['parameter'] = params
    config['name'] = 'ac_all_500'
    config['script'] = 'acrobot.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/acrobot_experiment_config_ac_all_500.json', 'wb' ) as fp:
        json.dump( config, fp )

def create_ac_all_1000():

    config = {}

    params = {}
    params['-e'] = [1000]
    params['-i'] = [1500]
    params['-R'] = [1] * 20
    params['-a'] = [.1]
    params['-b'] = [.005]
    params['-A'] = ['Q', 'GQ', 'PGQ']
    params['--behaviorTemperature'] = [1.1]
    params['--targetTemperature'] = [.5]


    config['parameter'] = params
    config['name'] = 'ac_all_1000'
    config['script'] = 'acrobot.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/acrobot_experiment_config_ac_all_1000.json', 'wb' ) as fp:
        json.dump( config, fp )

create_ac_all()
create_ac_all2()
create_ac_GQ2_PGQ2()
create_ac_all_500()
create_ac_all_1000()


