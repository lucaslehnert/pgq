'''
Created on Oct 17, 2015

@deprecated: Delete module.

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''
import numpy as np
import os
from mdp import MDPTabular
from basisfunction import tiledStateActionBinaryBasisFunction
from policy import BoltzmannPolicy
from qlearning import Q, GQ, PGQ
from experiment import evaluateQlearningSimulated
import matplotlib.pyplot as plt
from util.numpy_json import numpyDictionaryToJson, loadJSONResults

experimentDir = '../experiments/chainmdp/'
import os
if not os.path.exists( experimentDir ):
    os.makedirs( experimentDir )

def createChainMDP( length, gamma, transitionEpsilon=0.05 ):
    S = np.array( [range( length )] ).T
    A = ['left', 'right']
    def t( s, a, nexts ):
        prob = 0.0
        if a == 'right':
            if nexts - s == 1 or ( nexts == S[-1] and s == S[-1] ):
                prob = 1.0 - transitionEpsilon
            elif nexts - s == -1 or ( nexts == S[0] and s == S[0] ):
                prob = transitionEpsilon
        elif a == 'left':
            if nexts - s == -1 or ( nexts == S[0] and s == S[0] ):
                prob = 1.0 - transitionEpsilon
            elif nexts - s == 1 or ( nexts == S[-1] and s == S[-1] ):
                prob = transitionEpsilon
        else:
            raise Exception( 'Action ' + str( a ) + ' is not in the action space.' )
        return prob
    def r( s, a, nexts ):
        if a == 'right' and nexts == S[-1]:
            return 1.0
        elif a == 'left' and nexts == S[0]:
            return -1.0
        else:
            return 0.0

    startDistribution = np.zeros( len( S ) )
    startDistribution[0] = 1.0

    mdp = MDPTabular( stateSpace=S, actionSpace=A, transitionFunction=t, rewardFunction=r, \
                      gamma=gamma, startDistribution=startDistribution, goalState=S[-1] )
    return mdp

def getBasisFunction( mdp, tileSize ):
    return tiledStateActionBinaryBasisFunction( mdp.getStateSpace(), mdp.getActionSpace(), tileSize, 0 )

def demoBoltzmann():

    plotDir = '../plot/chainmdp/demo-q-boltzmanntarget/'
    if not os.path.exists( plotDir ):
        os.makedirs( plotDir )

    mdp = createChainMDP( 10, 0.9 )
    phi = getBasisFunction( mdp, tileSize=1 )

    behaviorTemp = 2.0
    targetTemp = 0.8
    piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=behaviorTemp )
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=targetTemp )

    initTheta = np.zeros( len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) )

    alpha = 0.05
    q = Q( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    resQ = evaluateQlearningSimulated( 20, q, mdp, phi, piBehavior, piTarget )

    alpha = 0.05
    beta = 0.25
    gq = GQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    resGQ = evaluateQlearningSimulated( 20, gq, mdp, phi, piBehavior, piTarget )

    alpha = 0.05
    beta = 0.25
    pgq = PGQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
               alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
               behaviorPolicy=piBehavior, targetPolicy=piTarget )
    resPGQ = evaluateQlearningSimulated( 20, pgq, mdp, phi, piBehavior, piTarget, max_episode_length=200 )

    print  np.sqrt( resPGQ['mspbeEp'] )

    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( resQ['mspbeEp'] ) ), np.sqrt( resQ['mspbeEp'] ), label='Q(0)', linewidth=2.0 )
    plt.plot( range( len( resGQ['mspbeEp'] ) ), np.sqrt( resGQ['mspbeEp'] ), label='GQ(0)', linewidth=2.0 )
    plt.plot( range( len( resPGQ['mspbeEp'] ) ), np.sqrt( resPGQ['mspbeEp'] ), label='PGQ(0)', linewidth=2.0 )
#    plt.ylim( [0, 200] )
#    plt.gca().set_xscale( 'log' )
    plt.xlabel( 'episodes' )
    plt.ylabel( 'Root MSPBE' )
    plt.legend()
    plt.show()

def runExperiment( **configuration ):
    print 'Running chainmdp experiment.'

    mdp = createChainMDP( 10, 0.99 )
    phi = getBasisFunction( mdp, 1 )
    initTheta = np.zeros( len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) )

    piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=configuration['behaviorTemperature'] )
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=configuration['targetTemperature'] )

    alpha = configuration['alpha']
    beta = configuration['beta']

    if configuration['agent'] == 'Q':
        agent = Q( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    elif configuration['agent'] == 'GQ':
        agent = GQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    elif configuration['agent'] == 'PGQ':
        agent = PGQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )

    repeatData = []
    for repeat in range( configuration['repeat'] ):
        print 'Running repeat ' + str( repeat )

        res = evaluateQlearningSimulated( configuration['episodes'], agent, mdp, phi, piBehavior, \
                                             piTarget, max_episode_length=200 )
        repeatData.append( res )

    return repeatData

def getParameterConfigurationList( **params ):
    '''
    Create a list of parameter configurations for which the experiment should be run.
    
    @param sweeps: Experiment repeats.
    @param behaviorPolicy: 
    @param targetPolicy: 
    @param behaviorTemperature: The control temperature of the Boltzmann control policy.
    @param targetTemperature: The target temperature of the Boltzmann target policy.
    @param alpha: The alpha learning rates.
    @param beta: The beta learning rates.
    @param agent: The algorithm names.
    
    @return: A list of parameter configurations.
    '''
    parameter = {}

    parameter['episodes'] = [params.get( 'episodes', 50 )]
    parameter['repeat'] = [params.get( 'repeat', 20 )]

    parameter['behaviorTemperature'] = params.get( 'behaviorTemperature', [2.0] )
    parameter['targetTemperature'] = params.get( 'targetTemperature', [0.2] )

    parameter['alpha'] = params.get( 'alpha', [0.2] )
    parameter['beta'] = params.get( 'beta', [0.5] )

    parameter['agent'] = params.get( 'algorithmName', ['Q', 'GQ', 'PGQ'] )

    from sklearn.grid_search import ParameterGrid
    parameterList = list( ParameterGrid( parameter ) )

    return parameterList

def generateConfigs_test():
    global experimentDir

    configList = getParameterConfigurationList( episodes=5, repeat=3, behaviorTemperature=[2.0], targetTemperature=[0.2],
                                                alpha=[0.05], beta=[0.25], algorithmName=['Q', 'GQ', 'PGQ'] )

    fileList = []
    for i in range( len( configList ) ):
        fileName = experimentDir + '/test_param_' + ( '%04d' % i ) + '.json'
        fileList.append( fileName )
        numpyDictionaryToJson( configList[i], fileName )
    print 'Generated ' + str( len( configList ) ) + ' configurations.'
    return fileList

def generateConfigs_m0():
    global experimentDir

    alphaList = np.append( np.logspace( -3, -2, 2 ), np.linspace( 0.1, 1.0, 10 ) )
    betaList = np.append( np.logspace( -3, -2, 2 ), np.linspace( 0.1, 1.0, 10 ) )
    behaviorTemp = np.array( [ 0.05, 0.1, 0.7, 2.0 ] )
    targetTemp = np.array( [ 0.05, 0.1, 0.7, 2.0 ] )

    configList = getParameterConfigurationList( episodes=50, repeat=20, behaviorTemperature=behaviorTemp, targetTemperature=targetTemp,
                                                alpha=alphaList, beta=[0.25], algorithmName=['Q'] )
    configList += getParameterConfigurationList( episodes=50, repeat=20, behaviorTemperature=behaviorTemp, targetTemperature=targetTemp,
                                                alpha=alphaList, beta=betaList, algorithmName=['GQ', 'PGQ'] )
    configList = filter( lambda c: c['behaviorTemperature'] >= c['targetTemperature'], configList )

    fileList = []
    for i in range( len( configList ) ):
        fileName = experimentDir + '/00_param_' + ( '%04d' % i ) + '.json'
        fileList.append( fileName )
        numpyDictionaryToJson( configList[i], fileName )
    print 'Generated ' + str( len( configList ) ) + ' configurations.'
    return fileList

def main():
    global experimentDir

    import argparse
    parser = argparse.ArgumentParser( description='Chain MDP Q Learning Experiments', \
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-c', '--config', action='store_true', \
                         help='Generate configuration .json files.' )
    parser.add_argument( '-i', '--experimentId', type=str, default='test', \
                         help='Experiment ID.' )
    parser.add_argument( '-p', '--parameterFile', type=str, nargs='*', \
                         help='Experiment parameter files to run.' )
    args = parser.parse_args()

    if args.config:
        print 'Generating config files for id ' + str( args.experimentId )
        if args.experimentId == 'test':
            fileList = generateConfigs_test()
        elif args.experimentId == '00':
            fileList = generateConfigs_m0()
    if args.parameterFile != None:
        if len( args.parameterFile ) > 0:
            parameterFileList = args.parameterFile
        else:
            parameterFileList = fileList

        for paramFile in parameterFileList:
            print 'Running experiment configuration ' + str( paramFile )
            configuration = loadJSONResults( paramFile )
            results = runExperiment( **configuration )
            res = {'configuration' : configuration,
                   'experiment' : 'baird',
                   'results' : results }

            resultFile = resultFileName( paramFile )
            numpyDictionaryToJson( res, resultFile )

    print 'Done'

if __name__ == '__main__':
    demoBoltzmann()
#    main()
    pass
