'''
Created on Jan 11, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

Control on the Acrobot domain.

'''
import numpy as np
import matplotlib.pyplot as plt

from mdp import MDPContinuousState
from basisfunction import getTiledStateActionBasisFunction
from policy import BoltzmannPolicy
from qlearning import Q, GQ, PGQ, SARSA
from experiment import experimentSimulateTransitions
from util.numpy_json import numpyDictionaryToJson

def rk4( derivatives, x_t, t, stepSize ):
    k1 = derivatives( x_t, t )
    k2 = derivatives( x_t + stepSize / 2.0 * k1, t + stepSize / 2.0 )
    k3 = derivatives( x_t + stepSize / 2.0 * k2, t + stepSize / 2.0 )
    k4 = derivatives( x_t + stepSize * k3, t + stepSize )

    x_t1 = x_t + stepSize / 6.0 * ( k1 + 2 * k2 + 2 * k3 + k4 )
    return x_t1

def wrap( x, m, M ):
    '''
    also from rlpy
    '''
    diff = M - m
    while x > M:
        x -= diff
    while x < m:
        x += diff
    return x

def bound( x, m, M ):
    return min( max( x, m ), M )

def createAcrobotMDP():

    thetaRange = np.linspace( 0, 2 * np.pi, 10 )
    thetaDot1Range = np.linspace( -4 * np.pi, 4 * np.pi, 9 )
    thetaDot2Range = np.linspace( -9 * np.pi, 9 * np.pi, 19 )
    t1, t2, t1d, t2d = np.meshgrid( thetaRange, thetaRange, thetaDot1Range, thetaDot2Range )
    discretizedStateSpace = np.array( [ t1.flatten(), t2.flatten(), t1d.flatten(), t2d.flatten() ], \
                                      dtype=np.double ).T

    torque = np.array( [ -1, 0, 1 ] )

    m1 = m2 = 1
    l1 = 1
    lc1 = lc2 = 0.5
    I1 = I2 = 1
    g = 9.8

    def derivatives( sa, t ):
        theta1 = sa[0]
        theta2 = sa[1]
        dtheta1 = sa[2]
        dtheta2 = sa[3]
        tau = sa[4]

        d1 = m1 * lc1 ** 2 + m2 \
             * ( l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos( theta2 ) ) + I1 + I2
        d2 = m2 * ( lc2 ** 2 + l1 * lc2 * np.cos( theta2 ) ) + I2
        phi2 = m2 * lc2 * g * np.cos( theta1 + theta2 - np.pi / 2. )
        phi1 = -m2 * l1 * lc2 * dtheta2 ** 2 * np.sin( theta2 ) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin( theta2 )  \
               + ( m1 * lc1 + m2 * l1 ) * g * np.cos( theta1 - np.pi / 2 ) + phi2
        ddtheta2 = ( tau + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin( theta2 ) - phi2 ) \
                / ( m2 * lc2 ** 2 + I2 - d2 ** 2 / d1 )
        ddtheta1 = -( d2 * ddtheta2 + phi1 ) / d1

        return np.array( [ dtheta1, dtheta2, ddtheta1, ddtheta2, 0. ] )

    def transitionSampler( state, action ):

        sa = np.append( state, [action] )
        sa_next = rk4( derivatives, sa, 0, .2 )

        staten = sa_next[:4]
        staten[0] = wrap( staten[0], -np.pi, np.pi )
        staten[1] = wrap( staten[1], -np.pi, np.pi )
        staten[2] = bound( staten[2], -4 * np.pi, 4 * np.pi )
        staten[3] = bound( staten[3], -9 * np.pi, 9 * np.pi )

        return staten

    isGoalState = lambda s :-np.cos( s[0] ) - np.cos( s[1] + s[0] ) > 1.
    def rewardFunction( state, action, staten ):
        return -1.0 if not isGoalState( staten ) else 0.0

    gamma = 1.0
    def startStateSampler():
        return np.zeros( 4 )
    discretizedStartStateDistribution = np.zeros( len( discretizedStateSpace ) )
    startInd = np.where( np.all( discretizedStateSpace == np.zeros( 4 ), axis=1 ) )[0][0]
    discretizedStartStateDistribution[startInd] = 1.0

    return MDPContinuousState( torque, transitionSampler, rewardFunction, gamma, startStateSampler, \
                               isGoalState, discretizedStateSpace, discretizedStartStateDistribution )

#def getBasisFunction( mdp ):
#    return tileCodedStateActionBinaryBasisFunction( \
#                    mdp.getStateSpace(), mdp.getActionSpace(), \
#                    [2. / 5. * np.pi, 2. / 5. * np.pi, 8. / 6. * np.pi, 18. / 6. * np.pi], 1 )

def getBasisFunction( mdp ):
    return getTiledStateActionBasisFunction( mdp, [12, 14, 12, 14] )

#    minS = np.array( map( lambda s: np.min( s ), mdp.getStateSpace().T ) )
#    maxS = np.array( map( lambda s: np.max( s ), mdp.getStateSpace().T ) )
#
##     tileNum = np.array([2.,2.,2.,2.])
#    tileLen = ( maxS - minS ) / tileNum
##     phiLen = int( np.prod(tileNum) )
##     print 'number of features: ' + str(phiLen)
#
#    def phis( s ):
#        stripeInd = np.array( np.floor( ( s - minS ) / tileLen - ( 10 ** -10 ) ), dtype=np.int )
#        phiv = []
#
#        for i in range( len( tileNum ) ):
#            stripe = np.zeros( tileNum[i] )
#            stripe[stripeInd[i]] = 1.0
#    #         print str(i) + ':' + str(stripe)
#            if len( phiv ) == 0:
#                phiv = stripe
#            else:
#                phiv = np.outer( phiv, stripe ).flatten()
#        return phiv
#
#    actionSet = mdp.getActionSpace()
#    def phia( a ):
#        aInd = np.where( a == actionSet )[0][0]
#        phiv = np.zeros( len( actionSet ) )
#        phiv[aInd] = 1.0
#        return phiv
#
#    def phi( s, a ):
#        ps = phis( s )
#        pa = phia( a )
#        return np.outer( pa, ps ).flatten()
#
#    return phi

def testAcrobot():

    mdp = createAcrobotMDP()
#    phi = getBasisFunction( mdp )
#    phi = tileCodedStateActionBinaryBasisFunction( \
#                    mdp.getStateSpace(), mdp.getActionSpace(), \
#                    [2. / 5. * np.pi, 2. / 5. * np.pi, 8. / 6. * np.pi, 18. / 6. * np.pi], 4 )
    phi = getBasisFunction( mdp )

    alpha = 0.05
    beta = 0.01
#    traceLambda = 0.0
    temp = 0.5
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


    iterations = 5000

    def rewardListener( s, a, snext, reward ):
        return reward
    tnorm = lambda t : np.linalg.norm( t.getTheta() )

    thetaNormLog = []
    episodeLengthLog = []
    theta = agent.getTheta()
    for epInd in range( 800 ):
        print 'Episode ' + str( epInd + 1 )
        errorBenchmarksRep, rewardLog = experimentSimulateTransitions( \
                                    iterations, mdp, controlPolicy, agent, \
                                    errorMeasures=[tnorm], transitionListener=[rewardListener], actionFromAgent=False )
        thetaNorm = map( lambda n: n[-1], errorBenchmarksRep )
        episodeLength = map( lambda e: len( e ), rewardLog )

        thetaNext = agent.getTheta()
        print theta - thetaNext
        theta = thetaNext

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
    plt.ylim( [0, 2000] )
    plt.show()

def runExperiment( **configuration ):

    mdp = createAcrobotMDP()
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
    tnorm = lambda t : np.linalg.norm( t.getTheta() )

    thetaNormExp = []
    episodeLengthExp = []

    actionFromAgent = True if configuration['agent'] == 'SARSA' else False

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
        try:
            thetaNormLog = []
            episodeLengthLog = []
            for _ in range( configuration['episodes'] ):
                errorBenchmarksRep, rewardLog, _ = experimentSimulateTransitions( \
                                            iterations, mdp, piBehavior, agent, \
                                            errorMeasures=[tnorm], transitionListener=[rewardListener], \
                                            actionFromAgent=actionFromAgent )
                thetaNorm = map( lambda n: n[-1], errorBenchmarksRep )
                episodeLength = map( lambda e: len( e ), rewardLog )

                thetaNormLog.append( thetaNorm[0] )
                episodeLengthLog.append( episodeLength[0] )
            thetaNormLog = np.array( thetaNormLog )
            episodeLengthLog = np.array( episodeLengthLog )

            thetaNormExp.append( thetaNormLog )
            episodeLengthExp.append( episodeLengthLog )

            successfulRepeats += 1
        except Exception as e:
            print e
            continue

    experimentResults = { 'thetaNorm' : thetaNormExp, 'episodeLength' : episodeLengthExp, \
                         'successfulRepeats' : successfulRepeats }
    return experimentResults


def main():

    import datetime
    startTime = datetime.datetime.now()
    print 'Started at ' + str( startTime )

    import argparse
    parser = argparse.ArgumentParser( description='Acrobot Experiments', \
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
