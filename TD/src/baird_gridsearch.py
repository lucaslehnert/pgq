'''
Created on Apr 24, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

Script to generate configurations for Baird counter example experiment.

'''

import json

import os
experimentDir = '../experiment/baird'
if not os.path.exists( experimentDir ):
    os.makedirs( experimentDir )

def create_bd_all():

    config = {}

    params = {}
    params['-i'] = [500]
    params['-R'] = [20]
    params['-a'] = [.01, .1]
    params['-b'] = [.1]
    params['-A'] = ['Q', 'GQ', 'GQ2', 'PGQ', 'PGQ2']
    params['-u'] = ['simulated']
    params['--behaviorPolicy'] = ['Boltzmann']
    params['--targetPolicy'] = ['Boltzmann']
    params['--behaviorTemperature'] = [10.0]
    params['--targetTemperature'] = [.2]


    config['parameter'] = params
    config['name'] = 'bd_all'
    config['script'] = 'baird.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/baird_experiment_config_bd_all.json', 'wb' ) as fp:
        json.dump( config, fp )

def create_bd_GQ2_PGQ2():

    config = {}

    params = {}
    params['-i'] = [500]
    params['-R'] = [20]
    params['-a'] = [.005, .01, .05, .1, .2, .3]
    params['-b'] = [.005, .01, .05, .1, .2, .3]
    params['-A'] = ['GQ2', 'PGQ2']
    params['-u'] = ['simulated']
    params['--behaviorPolicy'] = ['Boltzmann']
    params['--targetPolicy'] = ['Boltzmann']
    params['--behaviorTemperature'] = [10.0]
    params['--targetTemperature'] = [.2]


    config['parameter'] = params
    config['name'] = 'bd_GQ2_PGQ2'
    config['script'] = 'baird.py'
    config['resultDir'] = experimentDir
    config['logDir'] = experimentDir

    with open( experimentDir + '/baird_experiment_config_bd_GQ2_PGQ2.json', 'wb' ) as fp:
        json.dump( config, fp )

create_bd_all()
create_bd_GQ2_PGQ2()


