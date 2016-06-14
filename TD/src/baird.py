'''
Created on Sep 18, 2015

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

Control on the Baird Counter example.

'''
import numpy as np
from mdp import MDPTabular, mspbeStateActionValues, \
    mstdeStateActionValues
from basisfunction import basisFunctionMatrixStateAction
from qlearning import argmaxrand, GreedyGQ, GreedyQ, Q, GQ, PGQ
from policy import TabularProbabilityPolicy, GreedyPolicy, BoltzmannPolicy
import matplotlib.pyplot as plt
from experiment import evaluateQlearningDP, experimentSampleTransitions, \
    experimentDynamicProgrammingSweeps, experimentSimulateTransitions
from util.numpy_json import numpyDictionaryToJson
import os

def createBairdMDP( ds=None ):
    S = np.array( [range( 1, 8 )] ).T
    A = np.array( ['solid', 'dotted'] )
    def r( s, a, nexts ):
        return 0.0
    def t( s, a, nexts ):
        if a == 'solid':
            if nexts[0] == 7:
                return 1.0
            else:
                return 0.0
        elif a == 'dotted':
            if nexts[0] < 7:
                return 1.0 / 6.0
            else:
                return 0.0
    gamma = 0.99

    if ds is None:
        ds = np.ones( len( S ), dtype=np.double ) / float( len( S ) )
    mdp = MDPTabular( stateSpace=S, actionSpace=A, transitionFunction=t, rewardFunction=r, \
                      gamma=gamma, startDistribution=ds )
    return mdp

def testBairdMDP():
    mdp = createBairdMDP()
    print 'Discount factor: ' + str( mdp.getGamma() )
    print 'State space: ' + str( mdp.getStateSpace().T )
    print 'Action space: ' + str( mdp.getActionSpace() )
    action = 'solid'
    Ta = mdp.getTransitionMatrix( action )
    print 'Transition model of ' + str( action )
    for r in Ta:
        print r
    action = 'dotted'
    Ta = mdp.getTransitionMatrix( action )
    print 'Transition model of ' + str( action )
    for r in Ta:
        print r

    print 'Transition test cases:'
    action = 'dotted'
    for state in mdp.getStateSpace():
        print str( state ) + ',' + str( action ) + ': ' + str( mdp.getNextStateDistribution( state, action ) )
        print 'expected reward: ' + str( mdp.getRewardExpected( state, action ) )

    action = 'solid'
    for state in mdp.getStateSpace():
        print str( state ) + ',' + str( action ) + ': ' + str( mdp.getNextStateDistribution( state, action ) )
        print 'expected reward: ' + str( mdp.getRewardExpected( state, action ) )

def getBasisFunction( mdp ):
    def phi( s, a ):
        phi_s = np.zeros( 8 )
        if s[0] == 7:
            phi_s[0] = 2.0
            phi_s[7] = 1.0
        else:
            phi_s[0] = 1.0
            phi_s[int( s[0] )] = 2.0

        phi_sa = np.zeros( 16 )
        aInd = np.where( mdp.getActionSpace() == a )[0][0]
        phi_sa[aInd * 8:( aInd + 1 ) * 8] = phi_s
        return phi_sa

    def phi2( s, a ):
        phi_sa = np.zeros( 15 )
        if a == 'solid':
            if s[0] == 7:
                phi_sa[0] = 2.0
                phi_sa[7] = 1.0
            else:
                phi_sa[0] = 1.0
                phi_sa[int( s[0] )] = 2.0
        elif a == 'dotted':
            sind = np.where( mdp.getStateSpace() == s )[0][0]
            phi_sa[sind + 8] = 1.0
        return phi_sa

    return phi2

def testBasisFunction():
    mdp = createBairdMDP()
    phi = getBasisFunction( mdp )

    phiMat = basisFunctionMatrixStateAction( mdp, phi )
    for p in phiMat:
        print p

    for s, a in mdp.getStateActionPairIterable():
        print str( s ) + ',' + str( a ) + ':\t' + str( phi( s, a ) )

def testQLearningArgmaxrand():
    for _ in range( 10 ):
        print argmaxrand( np.array( [ 0, 1, 1, 0, 1 ] ) )

def getInitialTheta():
    initTheta = np.ones( 15 )
    initTheta[7] = 10.0
#    initTheta[15] = -1.0
    return initTheta

def demoGreedyQ():

    plotDir = '../plot/baird/demo-greedy-q/'
    if not os.path.exists( plotDir ):
        os.makedirs( plotDir )

    mdp = createBairdMDP()
    phi = getBasisFunction( mdp )
    piBehavior = TabularProbabilityPolicy( np.array( [1. / 7., 6. / 7.] ), mdp.getActionSpace() )
    piTarget = GreedyPolicy( mdp.getActionSpace() )
    initTheta = getInitialTheta()

    alpha = 0.01
    greedyQ = GreedyQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior )
    mspbeListQ, mstdeListQ, _ = evaluateQlearningDP( 50, greedyQ, mdp, phi, piBehavior, piTarget )


    alpha = 0.05
    beta = 0.25
    greedyGQ = GreedyGQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior )
    mspbeListGQ, mstdeListGQ, _ = evaluateQlearningDP( 50, greedyGQ, mdp, phi, piBehavior, piTarget )

    print 'RMSPBE GQ: ' + str( np.sqrt( mspbeListGQ ) )

    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( mspbeListQ ) ), np.sqrt( mspbeListQ ), label='Greedy-Q(0)', linewidth=2.0 )
    plt.plot( range( len( mspbeListGQ ) ), np.sqrt( mspbeListGQ ), label='Greedy-GQ(0)', linewidth=2.0 )
    plt.ylim( [0, 100] )
#    plt.gca().set_xscale( 'log' )
    plt.xlabel( 'sweeps' )
    plt.ylabel( 'Root MSPBE' )
    plt.legend()
#    plt.show()
    plt.savefig( plotDir + 'MSPBE_onesdistr.pdf' )

    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( mstdeListQ ) ), np.sqrt( mstdeListQ ), label='Greedy-Q(0)', linewidth=2.0 )
    plt.plot( range( len( mstdeListGQ ) ), np.sqrt( mstdeListGQ ), label='Greedy-GQ(0)', linewidth=2.0 )
    plt.ylim( [0, 50] )
#    plt.gca().set_xscale( 'log' )
    plt.xlabel( 'sweeps' )
    plt.ylabel( 'Root MSTDE' )
    plt.legend()
#    plt.show()
    plt.savefig( plotDir + 'MSTDE_onesdistr.pdf' )

def demoQEvaluation():

    plotDir = '../plot/baird/demo-q-evaluation/'
    if not os.path.exists( plotDir ):
        os.makedirs( plotDir )

    mdp = createBairdMDP()
    phi = getBasisFunction( mdp )
    piBehavior = TabularProbabilityPolicy( np.array( [1. / 7., 6. / 7.] ), mdp.getActionSpace() )
    piTarget = TabularProbabilityPolicy( np.array( [1., 0.] ), mdp.getActionSpace() )
    initTheta = getInitialTheta()

    alpha = 0.01
    q = Q( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    mspbeListQ, mstdeListQ, _ = evaluateQlearningDP( 50, q, mdp, phi, piBehavior, piTarget )


    alpha = 0.05
    beta = 0.25
    gq = GQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    mspbeListGQ, mstdeListGQ, _ = evaluateQlearningDP( 50, gq, mdp, phi, piBehavior, piTarget )

    print 'RMSPBE GQ: ' + str( np.sqrt( mspbeListGQ ) )

    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( mspbeListQ ) ), np.sqrt( mspbeListQ ), label='Q(0)', linewidth=2.0 )
    plt.plot( range( len( mspbeListGQ ) ), np.sqrt( mspbeListGQ ), label='GQ(0)', linewidth=2.0 )
    plt.ylim( [0, 40] )
#    plt.gca().set_xscale( 'log' )
    plt.xlabel( 'sweeps' )
    plt.ylabel( 'Root MSPBE' )
    plt.legend()
#    plt.show()
    plt.savefig( plotDir + 'MSPBE_onesdistr_uniformtransitionMSPBE.pdf' )

    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( mstdeListQ ) ), np.sqrt( mstdeListQ ), label='Q(0)', linewidth=2.0 )
    plt.plot( range( len( mstdeListGQ ) ), np.sqrt( mstdeListGQ ), label='GQ(0)', linewidth=2.0 )
    plt.ylim( [0, 4] )
#    plt.gca().set_xscale( 'log' )
    plt.xlabel( 'sweeps' )
    plt.ylabel( 'Root MSTDE' )
    plt.legend()
#    plt.show()
    plt.savefig( plotDir + 'MSTDE_onesdistr.pdf' )

def demoQBoltzmannTarget():

    plotDir = '../plot/baird/demo-q-boltzmanntarget/'
    if not os.path.exists( plotDir ):
        os.makedirs( plotDir )

    mdp = createBairdMDP()
    phi = getBasisFunction( mdp )

    targetTemp = 0.3
    piBehavior = TabularProbabilityPolicy( np.array( [1. / 7., 6. / 7.] ), mdp.getActionSpace() )
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=targetTemp )
#    piTarget = TabularProbabilityPolicy( np.array( [1., 0.] ), mdp.getActionSpace() )
    initTheta = getInitialTheta()

    alpha = 0.01
    q = Q( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    mspbeListQ, mstdeListQ, _ = evaluateQlearningDP( 50, q, mdp, phi, piBehavior, piTarget )


    alpha = 0.05
    beta = 0.25
    gq = GQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    mspbeListGQ, mstdeListGQ, _ = evaluateQlearningDP( 50, gq, mdp, phi, piBehavior, piTarget )

    alpha = 0.05
    beta = 0.5
    pgq = PGQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    mspbeListPGQ, mstdeListPGQ, _ = evaluateQlearningDP( 50, pgq, mdp, phi, piBehavior, piTarget )

    print 'RMSPBE GQ: ' + str( np.sqrt( mspbeListGQ ) )

    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( mspbeListQ ) ), np.sqrt( mspbeListQ ), label='Q(0)', linewidth=2.0 )
    plt.plot( range( len( mspbeListGQ ) ), np.sqrt( mspbeListGQ ), label='GQ(0)', linewidth=2.0 )
    plt.plot( range( len( mspbeListPGQ ) ), np.sqrt( mspbeListPGQ ), label='PGQ(0)', linewidth=2.0 )
#    plt.ylim( [0, 40] )
#    plt.gca().set_xscale( 'log' )
    plt.xlabel( 'sweeps' )
    plt.ylabel( 'Root MSPBE' )
    plt.legend()
#    plt.show()
    plt.savefig( plotDir + 'MSPBE_onesdistr_uniformtransitionMSPBE_targetTemp' + str( targetTemp ) + '.pdf' )

    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( mstdeListQ ) ), np.sqrt( mstdeListQ ), label='Q(0)', linewidth=2.0 )
    plt.plot( range( len( mstdeListGQ ) ), np.sqrt( mstdeListGQ ), label='GQ(0)', linewidth=2.0 )
    plt.plot( range( len( mstdeListPGQ ) ), np.sqrt( mstdeListPGQ ), label='PGQ(0)', linewidth=2.0 )
#    plt.ylim( [0, 4] )
#    plt.gca().set_xscale( 'log' )
    plt.xlabel( 'sweeps' )
    plt.ylabel( 'Root MSTDE' )
    plt.legend()
#    plt.show()
    plt.savefig( plotDir + 'MSTDE_onesdistr_targetTemp' + str( targetTemp ) + '.pdf' )

def demoQBoltzmann():

    plotDir = '../plot/baird/demo-q-boltzmann/'
    if not os.path.exists( plotDir ):
        os.makedirs( plotDir )

    mdp = createBairdMDP()
    phi = getBasisFunction( mdp )

    sweeps = 200

    behaviorTemp = 0.4
    targetTemp = 0.4

#    behaviorTemp = 0.5
#    targetTemp = 0.1
#    piBehavior = TabularProbabilityPolicy( np.array( [1. / 7., 6. / 7.] ), mdp.getActionSpace() )
    piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=behaviorTemp )
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=targetTemp )
#    piTarget = TabularProbabilityPolicy( np.array( [1., 0.] ), mdp.getActionSpace() )
    initTheta = getInitialTheta()

#    alpha = 0.01
#    q = Q( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
#                             alpha=alpha, actionSpace=mdp.getActionSpace(), \
#                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
#    mspbeListQ, mstdeListQ, _ = evaluateQlearningDP( 50, q, mdp, phi, piBehavior, piTarget )


#    alpha = 0.05
#    beta = 0.25
#    gq = GQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
#                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
#                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
#    mspbeListGQ, mstdeListGQ, thetaListGQ = evaluateQlearningDP( sweeps, gq, mdp, phi, piBehavior, piTarget )

#    print 'Theta list GQ:'
#    for t in thetaListGQ:
#        print t

    alpha = 0.01
    beta = 0.25
    pgq = PGQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )
    mspbeListPGQ, mstdeListPGQ, _ = evaluateQlearningDP( sweeps, pgq, mdp, phi, piBehavior, piTarget )

#    print 'Theta list PGQ:'
#    for t in thetaListPGQ:
#        print t

#    print 'RMSPBE Q:   ' + str( np.sqrt( mstdeListQ ) )
#    print 'RMSPBE GQ:  ' + str( np.sqrt( mstdeListGQ ) )
    print 'RMSPBE PGQ: ' + str( np.sqrt( mstdeListPGQ ) )

    plt.figure( figsize=( 6, 5 ) )
#    plt.plot( range( len( mspbeListQ ) ), np.sqrt( mspbeListQ ), label='Q(0)', linewidth=2.0 )
#    plt.plot( range( len( mspbeListGQ ) ), np.sqrt( mspbeListGQ ), label='GQ(0)', linewidth=2.0 )
    plt.plot( range( len( mspbeListPGQ ) ), np.sqrt( mspbeListPGQ ), label='PGQ(0)', linewidth=2.0 )
    plt.ylim( [0, 50] )
#    plt.gca().set_xscale( 'log' )
    plt.xlabel( 'sweeps' )
    plt.ylabel( 'Root MSPBE' )
    plt.legend()
    plt.show()
#    plt.savefig( plotDir + 'MSPBE_statdistr_uniformtransitionMSPBE_targetTemp' + str( targetTemp ) + '_behaviorTemp' + str( behaviorTemp ) + '.pdf' )

    plt.figure( figsize=( 6, 5 ) )
#    plt.plot( range( len( mstdeListQ ) ), np.sqrt( mstdeListQ ), label='Q(0)', linewidth=2.0 )
#    plt.plot( range( len( mstdeListGQ ) ), np.sqrt( mstdeListGQ ), label='GQ(0)', linewidth=2.0 )
    plt.plot( range( len( mstdeListPGQ ) ), np.sqrt( mstdeListPGQ ), label='PGQ(0)', linewidth=2.0 )
#    plt.ylim( [0, 50] )
    plt.gca().set_yscale( 'log' )
    plt.xlabel( 'sweeps' )
    plt.ylabel( 'Root MSTDE' )
    plt.legend()
    plt.show()
#    plt.savefig( plotDir + 'MSTDE_statdistr_targetTemp' + str( targetTemp ) + '_behaviorTemp' + str( behaviorTemp ) + '.pdf' )

def demoPGQTemperatureInstability():

    mdp = createBairdMDP()
    phi = getBasisFunction( mdp )
    d_sa = np.ones( 14. ) / 14.

    behaviorTemp = 0.2
    targetTemp = 10.0

    piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=behaviorTemp )
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=targetTemp )
    initTheta = getInitialTheta()

    alpha = 0.01
    beta = 0.25
    pgq = PGQ( initTheta=initTheta, basisFunction=phi, gamma=mdp.getGamma(), \
                             alpha=alpha, beta=beta, actionSpace=mdp.getActionSpace(), \
                             behaviorPolicy=piBehavior, targetPolicy=piTarget )

    mspbe = lambda t : mspbeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_sa=d_sa )
    mstde = lambda t : mstdeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_sa=d_sa )

    errorBenchmarksPGQRep, _ = experimentSimulateTransitions( 100, mdp, piTarget, pgq, errorMeasures=[mspbe, mstde] )

    print errorBenchmarksPGQRep[0]

def experimentDynamicProgramming():
    plotDir = '../plot/baird/dp/'
    if not os.path.exists( plotDir ):
        os.makedirs( plotDir )

    d_s = np.ones( 7 ) / 7.
#    d_sa = np.ones( 14. ) / 14.

    mdp = createBairdMDP( d_s )
    phi = getBasisFunction( mdp )
    theta = getInitialTheta()

    piBehaviorTemp = 0.8
    piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=piBehaviorTemp )
    piTargetTemp = 0.8
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=piTargetTemp )

    mspbe = lambda t : mspbeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_s=d_s )
    mstde = lambda t : mstdeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_s=d_s )
    Phi = basisFunctionMatrixStateAction( phi, mdp )
    qnorm = lambda t : np.linalg.norm( np.dot( Phi, t.getTheta() ), ord=np.inf )

    alpha = 0.01
    beta = 0.5
    sweeps = 200

    pgqAgent = PGQ( basisFunction=phi, initTheta=theta, alpha=alpha, beta=beta, behaviorPolicy=piBehavior, \
                        targetPolicy=piTarget, actionSpace=mdp.getActionSpace(), gamma=mdp.getGamma() )
    errorBenchmarksPGQ = experimentDynamicProgrammingSweeps( sweeps, mdp, piTarget, pgqAgent, errorMeasures=[mspbe, mstde, qnorm] )

    gqAgent = GQ( basisFunction=phi, initTheta=theta, alpha=alpha, beta=beta, behaviorPolicy=piBehavior, \
                        targetPolicy=piTarget, actionSpace=mdp.getActionSpace(), gamma=mdp.getGamma() )
    errorBenchmarksGQ = experimentDynamicProgrammingSweeps( sweeps, mdp, piTarget, gqAgent, errorMeasures=[mspbe, mstde, qnorm] )

    print 'GQ:  Q,inf norm' + str( errorBenchmarksGQ[2] )
    print 'PGQ: Q,inf norm' + str( errorBenchmarksPGQ[2] )

    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( errorBenchmarksGQ[0] ) ), errorBenchmarksGQ[0], label='GQ(0)' )
    plt.plot( range( len( errorBenchmarksPGQ[0] ) ), errorBenchmarksPGQ[0], label='PGQ(0)' )
#    plt.plot( range( len( errorBenchmarksQ[0] ) ), errorBenchmarksQ[0], label='Q(0)' )
    plt.xlabel( 'update' )
    plt.ylabel( 'MSPBE' )
#    plt.gca().set_xscale( 'log' )
    plt.ylim( [0, 600] )
    plt.legend()
    plt.show()


    plt.figure( figsize=( 6, 5 ) )
    plt.plot( range( len( errorBenchmarksGQ[2] ) ), errorBenchmarksGQ[2], label='GQ(0)' )
    plt.plot( range( len( errorBenchmarksPGQ[2] ) ), errorBenchmarksPGQ[2], label='PGQ(0)' )
#    plt.plot( range( len( errorBenchmarksQ[0] ) ), errorBenchmarksQ[0], label='Q(0)' )
    plt.xlabel( 'update' )
    plt.ylabel( '||Q||_inf' )
#    plt.gca().set_xscale( 'log' )
    plt.ylim( [0, 20] )
    plt.legend()
    plt.show()

def experimentSampledGradient():

    plotDir = '../plot/baird/samplegrad/'
    if not os.path.exists( plotDir ):
        os.makedirs( plotDir )

    d_s = np.ones( 7 ) / 7.
    d_sa = np.ones( 14. ) / 14.

    mdp = createBairdMDP( d_s )
    phi = getBasisFunction( mdp )
    theta = getInitialTheta()

    piBehaviorTemp = 0.4
    piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=piBehaviorTemp )
    piTargetTemp = 0.7
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=piTargetTemp )

    mspbe = lambda t : mspbeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_sa=d_sa )
    mstde = lambda t : mstdeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_sa=d_sa )

    alpha = 0.005
    beta = 0.01

    repeats = 20
    iterations = 1000

    print 'Running PGQ...'
    errorBenchmarksPGQ = [ np.zeros( ( repeats, iterations + 1 ) ), np.zeros( ( repeats, iterations + 1 ) ) ]
    for rep in range( repeats ):
        pgqAgent = PGQ( basisFunction=phi, initTheta=theta, alpha=alpha, beta=beta, behaviorPolicy=piBehavior, \
                        targetPolicy=piTarget, actionSpace=mdp.getActionSpace(), gamma=mdp.getGamma() )
        errorBenchmarksPGQRep = experimentSampleTransitions( iterations, mdp, piTarget, pgqAgent, errorMeasures=[mspbe, mstde] )
        for i in range( len( errorBenchmarksPGQRep ) ):
            errorBenchmarksPGQ[i][rep, :] = errorBenchmarksPGQRep[i]
    errorBenchmarksPGQ = map( lambda m: np.mean( m, axis=0 ), errorBenchmarksPGQ )

    print 'Running GQ...'
    errorBenchmarksGQ = [ np.zeros( ( repeats, iterations + 1 ) ), np.zeros( ( repeats, iterations + 1 ) ) ]
    for rep in range( repeats ):
        gqAgent = GQ( basisFunction=phi, initTheta=theta, alpha=alpha, beta=beta, behaviorPolicy=piBehavior, \
                        targetPolicy=piTarget, actionSpace=mdp.getActionSpace(), gamma=mdp.getGamma() )
        errorBenchmarksGQRep = experimentSampleTransitions( iterations, mdp, piTarget, gqAgent, errorMeasures=[mspbe, mstde] )
        for i in range( len( errorBenchmarksGQRep ) ):
            errorBenchmarksGQ[i][rep, :] = errorBenchmarksGQRep[i]
    errorBenchmarksGQ = map( lambda m: np.mean( m, axis=0 ), errorBenchmarksGQ )

    print 'Running Q...'
    qiterations = np.min( [iterations, 800] )
    errorBenchmarksQ = [ np.zeros( ( repeats, qiterations + 1 ) ), np.zeros( ( repeats, qiterations + 1 ) ) ]
    for rep in range( repeats ):
        qAgent = Q( basisFunction=phi, initTheta=theta, alpha=alpha, beta=beta, behaviorPolicy=piBehavior, \
                        targetPolicy=piTarget, actionSpace=mdp.getActionSpace(), gamma=mdp.getGamma() )
        errorBenchmarksQRep = experimentSampleTransitions( qiterations, mdp, piTarget, qAgent, errorMeasures=[mspbe, mstde] )
        for i in range( len( errorBenchmarksQRep ) ):
            errorBenchmarksQ[i][rep, :] = errorBenchmarksQRep[i]
    errorBenchmarksQ = map( lambda m: np.mean( m, axis=0 ), errorBenchmarksQ )

    import datetime
    import time
    ts = time.time()
    st = datetime.datetime.fromtimestamp( ts ).strftime( '%Y-%m-%d_%H:%M:%S.%f' )

    resDict = {'experiment' : 'baird.experimentSampledGradient', 'alpha' : alpha, 'beta' : beta, 'd_s' : d_s, \
               'd_sa' : d_sa, 'behaviorTemp' : piBehaviorTemp, 'targetTemp' : piTargetTemp, \
               'benchmark' : {
                    'mspbe' : {'PGQ' : errorBenchmarksPGQ[0], 'GQ' : errorBenchmarksGQ[0], 'Q' : errorBenchmarksQ[0]}, \
                    'mstde' : {'PGQ' : errorBenchmarksPGQ[1], 'GQ' : errorBenchmarksGQ[1], 'Q' : errorBenchmarksQ[1]}}}
    numpyDictionaryToJson( resDict, plotDir + '/baird_' + st + '.json' )

    plt.figure( figsize=( 6, 4 ) )
    plt.plot( range( len( errorBenchmarksPGQ[0] ) ), errorBenchmarksPGQ[0], label='PGQ(0)' )
    plt.plot( range( len( errorBenchmarksGQ[0] ) ), errorBenchmarksGQ[0], label='GQ(0)' )
    plt.plot( range( len( errorBenchmarksQ[0] ) ), errorBenchmarksQ[0], label='Q(0)' )
    plt.xlabel( 'update' )
    plt.ylabel( 'MSPBE' )
    plt.gca().set_xscale( 'log' )
    plt.ylim( [0, 500] )
    plt.legend( loc=3 )
    plt.gcf().subplots_adjust( bottom=0.15 )
    plt.savefig( plotDir + 'mspbe_' + st + '.pdf' )
#    plt.show()

    plt.figure( figsize=( 6, 4 ) )
    plt.plot( range( len( errorBenchmarksPGQ[1] ) ), errorBenchmarksPGQ[1], label='PGQ(0)' )
    plt.plot( range( len( errorBenchmarksGQ[1] ) ), errorBenchmarksGQ[1], label='GQ(0)' )
    plt.plot( range( len( errorBenchmarksQ[1] ) ), errorBenchmarksQ[1], label='Q(0)' )
    plt.xlabel( 'update' )
    plt.ylabel( 'MSTDE' )
    plt.gca().set_xscale( 'log' )
    plt.ylim( [0, 60] )
    plt.legend( loc=3 )
    plt.gcf().subplots_adjust( bottom=0.15 )
    plt.savefig( plotDir + 'mstde_' + st + '.pdf' )
#    plt.show()

    print 'Done all experiments'

def experimentSimulatedTrajectories():

    plotDir = '../plot/baird/simulatedtraj/'
    if not os.path.exists( plotDir ):
        os.makedirs( plotDir )

    d_s = np.ones( 7 ) / 7.
    d_sa = np.ones( 14. ) / 14.

    mdp = createBairdMDP( d_s )
    phi = getBasisFunction( mdp )
    theta = getInitialTheta()

    piBehaviorTemp = 10.0
    piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=piBehaviorTemp )
#    piBehavior = TabularProbabilityPolicy( np.array( [1. / 7., 6. / 7.] ), mdp.getActionSpace() )
    piTargetTemp = 0.8
    piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=piTargetTemp )

    mspbe = lambda t : mspbeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_sa=d_sa )
    mstde = lambda t : mstdeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_sa=d_sa )

    alpha = 0.0125
    beta = 3.37500 * alpha

    repeats = 10
    iterations = 400

    print 'Running PGQ...'
    errorBenchmarksPGQ = [ np.zeros( ( repeats, iterations + 1 ) ), np.zeros( ( repeats, iterations + 1 ) ) ]
    for rep in range( repeats ):
        pgqAgent = PGQ( basisFunction=phi, initTheta=theta, alpha=alpha, beta=beta, behaviorPolicy=piBehavior, \
                        targetPolicy=piTarget, actionSpace=mdp.getActionSpace(), gamma=mdp.getGamma() )
        errorBenchmarksPGQRep = experimentSimulateTransitions( iterations, mdp, piBehavior, pgqAgent, errorMeasures=[mspbe, mstde] )
        for i in range( len( errorBenchmarksPGQRep ) ):
            errorBenchmarksPGQ[i][rep, :] = errorBenchmarksPGQRep[i]
    errorBenchmarksPGQ = map( lambda m: np.mean( m, axis=0 ), errorBenchmarksPGQ )

    print 'Running GQ...'
    errorBenchmarksGQ = [ np.zeros( ( repeats, iterations + 1 ) ), np.zeros( ( repeats, iterations + 1 ) ) ]
    for rep in range( repeats ):
        gqAgent = GQ( basisFunction=phi, initTheta=theta, alpha=alpha, beta=beta, behaviorPolicy=piBehavior, \
                        targetPolicy=piTarget, actionSpace=mdp.getActionSpace(), gamma=mdp.getGamma() )
        errorBenchmarksGQRep = experimentSimulateTransitions( iterations, mdp, piBehavior, gqAgent, errorMeasures=[mspbe, mstde] )
        for i in range( len( errorBenchmarksGQRep ) ):
            errorBenchmarksGQ[i][rep, :] = errorBenchmarksGQRep[i]
    errorBenchmarksGQ = map( lambda m: np.mean( m, axis=0 ), errorBenchmarksGQ )

    print 'Running Q...'
    errorBenchmarksQ = [ np.zeros( ( repeats, iterations + 1 ) ), np.zeros( ( repeats, iterations + 1 ) ) ]
    for rep in range( repeats ):
        qAgent = Q( basisFunction=phi, initTheta=theta, alpha=alpha, beta=beta, behaviorPolicy=piBehavior, \
                        targetPolicy=piTarget, actionSpace=mdp.getActionSpace(), gamma=mdp.getGamma() )
        errorBenchmarksQRep = experimentSimulateTransitions( iterations, mdp, piBehavior, qAgent, errorMeasures=[mspbe, mstde] )
        for i in range( len( errorBenchmarksQRep ) ):
            errorBenchmarksQ[i][rep, :] = errorBenchmarksQRep[i]
    errorBenchmarksQ = map( lambda m: np.mean( m, axis=0 ), errorBenchmarksQ )

    import datetime
    import time
    ts = time.time()
    st = datetime.datetime.fromtimestamp( ts ).strftime( '%Y-%m-%d_%H:%M:%S.%f' )

    resDict = {'experiment' : 'baird.experimentSampledGradient', 'alpha' : alpha, 'beta' : beta, 'd_s' : d_s, \
               'd_sa' : d_sa, 'behaviorTemp' : piBehaviorTemp, 'targetTemp' : piTargetTemp, \
               'benchmark' : {
                    'mspbe' : {'PGQ' : errorBenchmarksPGQ[0], 'GQ' : errorBenchmarksGQ[0], 'Q' : errorBenchmarksQ[0]}, \
                    'mstde' : {'PGQ' : errorBenchmarksPGQ[1], 'GQ' : errorBenchmarksGQ[1], 'Q' : errorBenchmarksQ[1]}}}
    numpyDictionaryToJson( resDict, plotDir + '/baird_' + st + '.json' )

    plt.figure( figsize=( 6, 4 ) )
    plt.plot( range( len( errorBenchmarksPGQ[0] ) ), errorBenchmarksPGQ[0], label='PGQ(0)' )
    plt.plot( range( len( errorBenchmarksGQ[0] ) ), errorBenchmarksGQ[0], label='GQ(0)' )
    plt.plot( range( len( errorBenchmarksQ[0] ) ), errorBenchmarksQ[0], label='Q(0)' )
    plt.xlabel( 'update' )
    plt.ylabel( 'MSPBE' )
#    plt.gca().set_xscale( 'log' )
    plt.ylim( [0, 600] )
    plt.gcf().subplots_adjust( bottom=0.15 )
    plt.legend()
    plt.savefig( plotDir + 'mspbe_' + st + '.pdf' )
#    plt.show()

    plt.figure( figsize=( 6, 4 ) )
    plt.plot( range( len( errorBenchmarksPGQ[1] ) ), errorBenchmarksPGQ[1], label='PGQ(0)' )
    plt.plot( range( len( errorBenchmarksGQ[1] ) ), errorBenchmarksGQ[1], label='GQ(0)' )
    plt.plot( range( len( errorBenchmarksQ[1] ) ), errorBenchmarksQ[1], label='Q(0)' )
    plt.xlabel( 'update' )
    plt.ylabel( 'MSTDE' )
#    plt.gca().set_xscale( 'log' )
    plt.ylim( [0, 80] )
    plt.gcf().subplots_adjust( bottom=0.15 )
    plt.legend()
    plt.savefig( plotDir + 'mstde_' + st + '.pdf' )
#    plt.show()

    print 'Done all experiments'

def runExperiment( **configuration ):
    print 'Running baird experiment'

    mdp = createBairdMDP()
    phi = getBasisFunction( mdp )
    initTheta = getInitialTheta()

    if configuration['behaviorPolicy'] == 'Baird':
        piBehavior = TabularProbabilityPolicy( np.array( [1. / 7., 6. / 7.] ), mdp.getActionSpace() )
    elif configuration['behaviorPolicy'] == 'Boltzmann':
        piBehavior = BoltzmannPolicy( mdp.getActionSpace(), temperature=configuration['behaviorTemperature'] )
    else:
        raise Exception( 'Unrecognized policy type.' )

    if configuration['targetPolicy'] == 'Baird':
        piTarget = TabularProbabilityPolicy( np.array( [1., 0.] ), mdp.getActionSpace() )
    elif configuration['targetPolicy'] == 'Boltzmann':
        piTarget = BoltzmannPolicy( mdp.getActionSpace(), temperature=configuration['targetTemperature'] )
    else:
        raise Exception( 'Unrecognized policy type.' )

    alpha = configuration['alpha']
    beta = configuration['beta']

    iterations = configuration['iterations']
    repeats = configuration['repeats']
    d_sa = np.ones( len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) ) / float( len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) )
    mspbe = lambda t : mspbeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_sa=d_sa )
    mstde = lambda t : mstdeStateActionValues( t.getTheta(), mdp, phi, piTarget, parametricPolicy=True, d_sa=d_sa )
    Phi = basisFunctionMatrixStateAction( phi, mdp )
    qnorm = lambda t : np.linalg.norm( np.dot( Phi, t.getTheta() ), ord=np.inf )

    mspbeList = []
    mstdeList = []
    qnormList = []
    mspbeListDiv = []
    mstdeListDiv = []
    qnormListDiv = []

    successfulRepeats = 0
    for rep in range( repeats ):
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

        print 'Running repeat ' + str( rep )
        if configuration['update'] == 'simulated':
            errorBenchmarksRep, _, completed = experimentSimulateTransitions( iterations, mdp, piBehavior, agent, \
                                                                   errorMeasures=[mspbe, mstde, qnorm] )
        elif configuration['update'] == 'sampled':
            errorBenchmarksRep, completed = experimentSampleTransitions( iterations, mdp, piBehavior, agent, \
                                                              errorMeasures=[mspbe, mstde, qnorm] )
        else:
            raise Exception( 'Unrecognized ' + str( configuration['update'] ) )


        if completed:
            mspbeList.append( errorBenchmarksRep[0] )
            mstdeList.append( errorBenchmarksRep[1] )
            qnormList.append( errorBenchmarksRep[2] )
        else:
            mspbeListDiv.append( errorBenchmarksRep[0] )
            mstdeListDiv.append( errorBenchmarksRep[1] )
            qnormListDiv.append( errorBenchmarksRep[2] )

        successfulRepeats += 1

#        continue
#        for i in range( len( errorBenchmarksRep ) ):
#            errorBenchmarks[i][rep, :] = errorBenchmarksRep[i]
#    mspbeList = errorBenchmarks[0]
#    mstdeList = errorBenchmarks[1]
#    qnormList = errorBenchmarks[2]

    episodeLog = { 'mspbe': mspbeList, 'mstde': mstdeList, 'qnorm': qnormList, \
                   'mspbeDiv': mspbeListDiv, 'mstdeDiv': mstdeListDiv, 'qnormDiv': qnormListDiv, \
                   'successfulRepeats' : successfulRepeats }
    return episodeLog

def main():

    import datetime
    startTime = datetime.datetime.now()
    print 'Started at ' + str( startTime )

    import argparse
    parser = argparse.ArgumentParser( description='Baird Experiments', \
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-r', '--resultFile', type=str, default='../experiment/test.json', help='Result file path.' )
    parser.add_argument( '-i', '--iterations', type=int, default=10, help='Number of iterations to run.' )
    parser.add_argument( '-R', '--repeats', type=int, default=1, help='Number of repeats to run.' )
    parser.add_argument( '-a', '--alpha', type=float, default=0.1, help='Alpha learning rate to run.' )
    parser.add_argument( '-b', '--beta', type=float, default=0.1, help='Beta learning rate to run.' )
    parser.add_argument( '-A', '--agent', type=str, default='GQ', help='Algorithm to run.' )
    parser.add_argument( '-u', '--update', type=str, default='simulated', help='Update method used.' )
    parser.add_argument( '--behaviorPolicy', type=str, default='Boltzmann', help='Behavior policy.' )
    parser.add_argument( '--targetPolicy', type=str, default='Boltzmann', help='Target policy.' )
    parser.add_argument( '--behaviorTemperature', type=float, default=1.0, help='Behavior temperature.' )
    parser.add_argument( '--targetTemperature', type=float, default=1.0, help='Target temperature.' )
    args = parser.parse_args()

    configuration = {}
    configuration['iterations'] = args.iterations
    configuration['repeats'] = args.repeats
    configuration['agent'] = args.agent
    configuration['alpha'] = args.alpha
    configuration['beta'] = args.beta
    configuration['update'] = args.update
    configuration['behaviorPolicy'] = args.behaviorPolicy
    configuration['targetPolicy'] = args.targetPolicy
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
