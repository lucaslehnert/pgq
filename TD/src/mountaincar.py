'''
Created on Jan 14, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

Mountain car implementation without noise.

'''
import numpy as np
import matplotlib.pyplot as plt
from mdp import MDPContinuousState
from basisfunction import getTiledStateActionBasisFunction
from util.numpy_json import numpyDictionaryToJson
from policy import BoltzmannPolicy
from qlearning import Q, GQ, PGQ, SARSA
from experiment import experimentSimulateTransitions

def bound( x, m, M ):
    return min( max( x, m ), M )

def createMountainCarMDP():

    positionMin = -1.2
    positionMax = 0.6
    velocityMin = -0.07
    velocityMax = 0.07

    positionGoal = 0.5

    ''' backward, neutral, forward '''
    actionSpace = np.array( [-1, 0, 1] )

    def transitionSampler( state, action ):
        pos = state[0]
        vel = state[1]

        pos = pos + vel
        pos = bound( pos, positionMin, positionMax )
        vel = vel + action * 0.001 - 0.0025 * np.cos( 3 * pos )
        vel = bound( vel, velocityMin, velocityMax )

        staten = np.array( [ pos, vel ] )
        return staten

    isGoalState = lambda s : s[0] >= positionGoal

    def rewardFunction( state, action, staten ):
        if isGoalState( staten ):
            return 0.0
        else:
            return -1.0

    def startStateSampler():
        return np.array( [ -0.5, 0.0 ] )

    gamma = 1.0

    statePosRange = np.linspace( positionMin, positionMax, 19 )
    stateVelRange = np.linspace( velocityMin, velocityMax, 11 )
    statePos, stateVel = np.meshgrid( statePosRange, stateVelRange )
    discretizedStateSpace = np.array( [statePos.flatten(), stateVel.flatten()], dtype=np.double ).T

    discretizedStartStateDistribution = np.zeros( len( discretizedStateSpace ) )
    startInd = np.where( np.all( discretizedStateSpace == np.array( [-0.5, 0.0] ), axis=1 ) )[0][0]
    discretizedStartStateDistribution[startInd] = 1.0

    return MDPContinuousState( actionSpace, transitionSampler, rewardFunction, gamma, startStateSampler, \
                               isGoalState, discretizedStateSpace, discretizedStartStateDistribution )

def getBasisFunction( mdp ):
    return getTiledStateActionBasisFunction( mdp, np.array( [18, 18] ) )

def testAcrobot():

    mdp = createMountainCarMDP()
#    phi = getBasisFunction( mdp )
#    phi = tileCodedStateActionBinaryBasisFunction( \
#                    mdp.getStateSpace(), mdp.getActionSpace(), \
#                    [2. / 5. * np.pi, 2. / 5. * np.pi, 8. / 6. * np.pi, 18. / 6. * np.pi], 4 )
    phi = getBasisFunction( mdp )

#    ''' This is good for sarsa '''
#    alpha = 0.05
#    traceLambda = 0.0
#    temp = 0.5

#    ''' This is good for Q-learning '''
#    alpha = 0.05
#    temp = 0.5

    ''' This is good for GQ '''
    alpha = 0.05
    beta = 0.05
    temp = 0.5

#    ''' This is good for PGQ '''
#    alpha = 0.05
#    beta = 0.1
#    temp = 0.5

    controlPolicy = BoltzmannPolicy( mdp.getActionSpace(), temperature=temp )
#    controlPolicy = GreedyPolicy( mdp.getActionSpace() )
    initTheta = np.zeros( len( phi( mdp.getStateSpace()[0], mdp.getActionSpace()[0] ) ) )
#    agent = SARSA( mdp.getGamma(), controlPolicy, traceLambda=traceLambda, basisFunction=phi, \
#                   initTheta=initTheta, alpha=alpha )
#    agent = Q( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
#                             alpha=alpha, actionSpace=mdp.getActionSpace(), \
#                             behaviorPolicy=controlPolicy, targetPolicy=controlPolicy )
    agent = GQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=controlPolicy, targetPolicy=controlPolicy )
#    agent = PGQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
#                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
#                             behaviorPolicy=controlPolicy, targetPolicy=controlPolicy )

    iterations = 20000

    def rewardListener( s, a, snext, reward ):
        return reward
    tnorm = lambda t : np.linalg.norm( t.getTheta() )

    thetaNormLog = []
    episodeLengthLog = []
    for epInd in range( 120 ):
        print 'Episode ' + str( epInd + 1 )
        errorBenchmarksRep, rewardLog, _ = experimentSimulateTransitions( iterations, mdp, controlPolicy, \
                                                                       agent, errorMeasures=[tnorm], \
                                                                       transitionListener=[rewardListener], \
                                                                       actionFromAgent=False )
        thetaNorm = map( lambda n: n[-1], errorBenchmarksRep )
        episodeLength = map( lambda e: len( e ), rewardLog )

        thetaNormLog.append( thetaNorm[0] )
        episodeLengthLog.append( episodeLength[0] )
        print '\tLength: ' + str( episodeLength[0] )
        print '\ttheta norm: ' + str( thetaNormLog[0] )

    thetaNormLog = np.array( thetaNormLog )
    episodeLengthLog = np.array( episodeLengthLog )

#    episodeLengthMean = np.mean( episodeLengthLog, axis=0 )
#    episodeLengthStd = np.std( episodeLengthLog, axis=0 )
#    episodeStep = range( 0, len( episodeLengthMean ) )
#
#    episodeLengthMeanSub = episodeLengthMean[0:len( episodeLengthMean ):50]
#    episodeLengthStdSub = episodeLengthStd[0:len( episodeLengthStd ):50]
#    episodeStepSub = episodeStep[0:len( episodeStep ):50]

    print episodeLengthLog
    plt.plot( range( 1, len( episodeLengthLog ) + 1 ), episodeLengthLog )
    plt.xlabel( 'Episode' )
    plt.ylabel( 'Episode Length' )
#    plt.gca().setyscale( 'log' )
    plt.ylim( [0, iterations] )
    plt.show()

def runExperiment( **configuration ):

    mdp = createMountainCarMDP()
    phi = getBasisFunction( mdp )
    initTheta = np.zeros( len( phi( mdp.getStateSpace()[0], mdp.getActionSpace()[0] ) ) )

    piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=configuration['behaviorTemperature'] )
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=configuration['targetTemperature'] )

    alpha = configuration['alpha']
    beta = configuration['beta']

    iterations = configuration['iterations']
    repeats = configuration['repeats']

    def rewardListener( s, a, snext, reward ):
        return reward
#    tnorm = lambda t : np.linalg.norm( t.getTheta() )
#    getTheta = lambda a: a.getTheta()

    actionFromAgent = True if configuration['agent'] == 'SARSA' else False

    lengthLog = []
    lengthLogFailed = []
    successfulRepeats = 0
    for rep in range( repeats ):

        if configuration['agent'] == 'Q':
            agent = Q( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                                 alpha=alpha, actionSpace=mdp.getActionSpace(), \
                                 behaviorPolicy=piBehavior, targetPolicy=piTarget )
        elif configuration['agent'] == 'SARSA':
            agent = SARSA( mdp.getGamma(), behaviorPolicy=piBehavior, traceLambda=configuration['traceLambda'], \
                           basisFunction=phi, initTheta=initTheta, alpha=alpha )
        elif configuration['agent'] == 'GQ':
            agent = GQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                                 alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                                 behaviorPolicy=piBehavior, targetPolicy=piTarget )
        elif configuration['agent'] == 'PGQ':
            agent = PGQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                                 alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                                 behaviorPolicy=piBehavior, targetPolicy=piTarget )

        print 'Running repeat ' + str( rep )

        episodeLengthLog = []
        episodeLengthFailed = []
        for _ in range( configuration['episodes'] ):
            _, rewardLog, completed = experimentSimulateTransitions( \
                                        iterations, mdp, piBehavior, agent, \
                                        errorMeasures=[], transitionListener=[rewardListener], \
                                        actionFromAgent=actionFromAgent )
            if completed:
                episodeLengthLog.append( len( rewardLog[0] ) )
            else:
                episodeLengthFailed.append( len( rewardLog[0] ) )
                break

#        print 'episodeLength size: ' + str( np.shape( episodeLengthLog ) )
        lengthLog.append( episodeLengthLog )
        lengthLogFailed.append( episodeLengthFailed )
        successfulRepeats += 1

    experimentResults = { 'episodeLength' : lengthLog, 'episodeLengthFailed' : lengthLogFailed, \
                          'successfulRepeats' : successfulRepeats }
    return experimentResults

def main():

    import datetime
    startTime = datetime.datetime.now()
    print 'Started at ' + str( startTime )

    import argparse
    parser = argparse.ArgumentParser( description='Mountain Car Experiments', \
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-r', '--resultFile', type=str, default='../experiment/test.json', help='Result file path.' )
    parser.add_argument( '-e', '--episodes', type=int, default=1, help='Number of episodes to run.' )
    parser.add_argument( '-i', '--iterations', type=int, default=10, help='Number of iterations to run.' )
    parser.add_argument( '-R', '--repeats', type=int, default=1, help='Number of repeats to run.' )
    parser.add_argument( '-a', '--alpha', type=float, default=0.1, help='Alpha learning rate to run.' )
    parser.add_argument( '-b', '--beta', type=float, default=0.1, help='Beta learning rate to run.' )
    parser.add_argument( '-A', '--agent', type=str, default='GQ', help='Algorithm to run.' )
    parser.add_argument( '--behaviorTemperature', type=float, default=1.0, help='Behavior temperature.' )
    parser.add_argument( '--targetTemperature', type=float, default=1.0, help='Target temperature.' )
    args = parser.parse_args()

    configuration = {}
    configuration['episodes'] = args.episodes
    configuration['iterations'] = args.iterations
    configuration['repeats'] = args.repeats
    configuration['agent'] = args.agent
    configuration['alpha'] = args.alpha
    configuration['beta'] = args.beta
    configuration['behaviorTemperature'] = args.behaviorTemperature
    configuration['targetTemperature'] = args.targetTemperature

    experimentResults = runExperiment( **configuration )
    result = {'configuration' : configuration,
              'experiment'    : 'mountaincar',
              'results'       : experimentResults }
    numpyDictionaryToJson( result, args.resultFile )

    stopTime = datetime.datetime.now()
    print 'Done at    ' + str( stopTime )
    print 'Durection  ' + str( stopTime - startTime )

if __name__ == '__main__':
    main()
    pass
