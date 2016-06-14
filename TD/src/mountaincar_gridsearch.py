'''
Created on Apr 24, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

Script to generate configurations for Baird counter example experiment.

'''

import json
import os
experimentDir = '../experiment/mountaincar'
if not os.path.exists( experimentDir ):
    os.makedirs( experimentDir )

def create_mc_GQ():

    config = {}

    params = {}
    params['-e'] = [10]
    params['-i'] = [12000]
    params['-R'] = [5]
    params['-a'] = [.1]
    params['-b'] = [.005]
    params['-A'] = ['GQ']
    params['--behaviorTemperature'] = [1.1]
    params['--targetTemperature'] = [.5]


    config['parameter'] = params
    config['name'] = 'mc'
    config['script'] = 'mountaincar.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/mountaincar_experiment_config.json', 'wb' ) as fp:
        json.dump( config, fp )

def create_mc_all():

    config = {}

    params = {}
    params['-e'] = [100]
    params['-i'] = [12000]
    params['-R'] = [20]
    params['-a'] = [.1]
    params['-b'] = [.005]
    params['-A'] = ['Q', 'GQ', 'GQ2', 'PGQ', 'PGQ2']
    params['--behaviorTemperature'] = [1.1]
    params['--targetTemperature'] = [.5]


    config['parameter'] = params
    config['name'] = 'mc_all'
    config['script'] = 'mountaincar.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/mountaincar_experiment_config_mc_all.json', 'wb' ) as fp:
        json.dump( config, fp )

def create_mc_all_500():

    config = {}

    params = {}
    params['-e'] = [500]
    params['-i'] = [12000]
    params['-R'] = [1] * 20
    params['-a'] = [.1]
    params['-b'] = [.005]
    params['-A'] = ['Q', 'GQ', 'PGQ']
    params['--behaviorTemperature'] = [1.1]
    params['--targetTemperature'] = [.5]


    config['parameter'] = params
    config['name'] = 'mc_all_500'
    config['script'] = 'mountaincar.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/mountaincar_experiment_config_mc_all_500.json', 'wb' ) as fp:
        json.dump( config, fp )

def create_mc_GQ2_PGQ2():

    config = {}

    params = {}
    params['-e'] = [100]
    params['-i'] = [12000]
    params['-R'] = [20]
    params['-a'] = [.001, .005, .01, .05, .1, .2]
    params['-b'] = [.001, .005, .01, .05, .1, .2]
    params['-A'] = ['GQ2', 'PGQ2']
    params['--behaviorTemperature'] = [1.1]
    params['--targetTemperature'] = [.5]


    config['parameter'] = params
    config['name'] = 'mc_GQ2_PGQ2'
    config['script'] = 'mountaincar.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/mountaincar_experiment_config_mc_GQ2_PGQ2.json', 'wb' ) as fp:
        json.dump( config, fp )

create_mc_GQ()
create_mc_all()
create_mc_all_500()
create_mc_GQ2_PGQ2()


