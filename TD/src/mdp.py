'''
Created on Sep 18, 2015

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

This module contains an implementation of a tabular and continuous state MDP classes.
It also contains different helper functions to compute distributions over state
spaces and MSPBE values.

'''

import numpy as np
from basisfunction import basisFunctionMatrixStateAction, \
    basisFunctionMatrixState

class MDPTabular( object ):
    '''
    classdocs
    '''


    def __init__( self, stateSpace, actionSpace, transitionFunction, rewardFunction, \
                  gamma=1.0, startDistribution=None, goalState=None ):
        '''
        Construct a tabular MDP class.
        
        @param stateSpace: State space.
        @param actionSpace: Action space.
        @param transitionFunction: Transition function.
        @param rewardFunction: Reward function.
        @param gamma: Discount factor.
        @param startDistribution: Start state distribution. If not specified, a unifrom
            distribution over states is assumed.
        @param goalState: Goal state of the MDP. If not specified, the MDP does not
            have a goal state.
        '''
        self.__stateSpace = np.array( stateSpace, copy=True )
        self.__stateSpace.flags.writeable = False
        self.__actionSpace = np.array( actionSpace, copy=True )
        self.__actionSpace.flags.writeable = False
        self.__gamma = gamma

        self.__transitionModel = self._constructTransitionMatrix( transitionFunction )
        self.__rewardMatrix = self._contructRewardMatrix( rewardFunction )

        if startDistribution is None:
            self.__startDistribution = np.ones( len( self.__stateSpace ) ) / float( len( self.__stateSpace ) )
        else:
            self.__startDistribution = np.array( startDistribution, copy=True )
        self.__startDistribution.flags.writeable = False

        self.__goalState = goalState

    def _constructTransitionMatrix( self, transitionFunction ):
        S = self.__stateSpace
        A = self.__actionSpace
        T = np.zeros( ( len( S ) * len( self.__actionSpace ), len( S ) ) )
        for si in range( len( S ) ):
            for ai in range( len( A ) ):
                for sj in range( len( S ) ):
                    T[ ai * len( S ) + si, sj  ] = transitionFunction( S[si], A[ai], S[sj] )
        T.flags.writeable = False
        return T

    def getTransitionMatrix( self, action ):
        ai = self.indexOfAction( action )
        return np.array( self.__transitionModel[ai * len( self.__stateSpace ):( ai + 1 ) * len( self.__stateSpace )], copy=False )

    def getNextStateDistribution( self, state, action ):
        Ta = self.getTransitionMatrix( action )
        si = self.indexOfState( state )
        ds = np.zeros( len( self.__stateSpace ) )
        ds[si] = 1.0
        nextStateDistr = np.dot( ds, Ta )
        return nextStateDistr

    def sampleNextState( self, state, action ):
        nextStateDistr = self.getNextStateDistribution( state, action )
        nextStateInd = np.argmax( np.random.multinomial( 1, nextStateDistr ) )
        nextState = self.__stateSpace[nextStateInd]
        return nextState

    def sampleStartState( self ):
        startStateInd = np.argmax( np.random.multinomial( 1, self.__startDistribution ) )
        startState = self.__stateSpace[startStateInd]
        return startState

    def startStateDistribution( self ):
        return np.array( self.__startDistribution, copy=False )

    def _contructRewardMatrix( self, rewardFunction ):
        S = self.__stateSpace
        A = self.__actionSpace
        R = np.zeros( ( len( S ) * len( self.__actionSpace ), len( S ) ) )
        for si in range( len( S ) ):
            for ai in range( len( A ) ):
                for sj in range( len( S ) ):
                    R[ ai * len( S ) + si, sj ] = rewardFunction( S[si], A[ai], S[sj] )
        return R

    def getReward( self, state, action, nextState ):
        si = self.indexOfState( state )
        ai = self.indexOfAction( action )
        sj = self.indexOfState( nextState )
        return self.__rewardMatrix[ ai * len( self.__stateSpace ) + si, sj ]

    def getRewardExpected( self, state, action ):
        si = self.indexOfState( state )
        ai = self.indexOfAction( action )
        rewards = self.__rewardMatrix[ ai * len( self.__stateSpace ) + si ]
        nextStateDistr = self.getNextStateDistribution( state, action )
        return np.dot( rewards, nextStateDistr )

    def indexOfState( self, state ):
        '''
        @return: Index of state in state space.
        '''
        indAr = np.where( np.all( self.getStateSpace() == state, axis=1 ) )
        if len( indAr ) == 0:
            raise Exception( 'Given state vector is not in state space. stateVector=' + str( state ) )
        return int( indAr[0] )

    def indexOfAction( self, action ):
        '''
        @return: Index of action in action space.
        '''
        return int( np.where( self.getActionSpace() == action )[0][0] )

    def indexOfStateActionPair( self, sapair ):
        '''
        Get the index of a state action pair.
        
        @param sapair: A pair (state, action)
        
        @return: Index of the pair in state-action space (as returned by getStateActionPairIterable)
        '''
        indState = self.indexOfState( sapair[0] )
        indAction = self.indexOfAction( sapair[1] )
        return indAction * len( self.getStateSpace() ) + indState

    def getStateActionPairIterable( self ):
        '''
        @return: An iterable object that iterates over all state action pairs.
        '''

        class SAIter:

            def __init__( self, stateSpace, actionSpace ):
                self.__stateSpace = stateSpace
                self.__actionSpace = actionSpace
                self.__ind = 0

            def next( self ):
                actInd = self.__ind / len( self.__stateSpace )
                stateInd = self.__ind % len( self.__stateSpace )
                self.__ind += 1
                if self.__ind > len( self.__stateSpace ) * len( self.__actionSpace ):
                    raise StopIteration
                return ( self.__stateSpace[stateInd], self.__actionSpace[actInd] )

        class SA:

            def __init__( self, stateSpace, actionSpace ):
                self.__stateSpace = stateSpace
                self.__actionSpace = actionSpace

            def __iter__( self ):
                return SAIter( self.__stateSpace, self.__actionSpace )

            def __len__( self ):
                return len( self.__stateSpace ) * len( self.__actionSpace )

            def __getitem__( self, i ):
                return self.__getslice__( i, i + 1 )[0]

            def __getslice__( self, i, j ):
                res = []
                for ind in range( i, j ):
                    actInd = ind / len( self.__stateSpace )
                    stateInd = ind % len( self.__stateSpace )
                    res.append( ( self.__stateSpace[stateInd], self.__actionSpace[actInd] ) )
                return res

        return SA( self.getStateSpace(), self.getActionSpace() )

    def isGoalState( self, stateVector ):
        return np.all( stateVector == self.__goalState )

    def getActionSpace( self ):
        return np.array( self.__actionSpace, copy=False )

    def getStateSpace( self ):
        return np.array( self.__stateSpace, copy=False )

    def getGamma( self ):
        return self.__gamma

class MDPContinuousState( object ):
    '''
    classdocs
    '''

    def __init__( self, actionSpace, nextStateSampler, rewardFunction, gamma, startStateSampler, isGoalState, \
                  discretizedStateSpace, discretizedStartStateDistribution ):
        self.__actionSpace = actionSpace
        self.__nextStateSampler = nextStateSampler
        self.__rewardFunction = rewardFunction
        self.__gamma = gamma
        self.__startStateSampler = startStateSampler
        self.__isGoalState = isGoalState
        self.__discretizedStateSpace = discretizedStateSpace
        self.__discretizedStartStateDistribution = discretizedStartStateDistribution

#    def getNextStateDistribution( self, state, action ):
#        return nextStateDistr

    def sampleStartState( self ):
        return self.__startStateSampler()

    def sampleNextState( self, state, action ):
        return self.__nextStateSampler( state, action )

#    def startStateDistribution( self ):
#        return np.array( self.__startDistribution, copy=False )

    def getReward( self, state, action, nextState ):
        return self.__rewardFunction( state, action, nextState )

#    def getRewardExpected( self, state, action ):
#        return 0.0

#    def indexOfState( self, state ):
#        '''
#        @return: Index of state in state space.
#        '''
#        indAr = np.where( np.all( self.getStateSpace() == state, axis=1 ) )
#        if len( indAr ) == 0:
#            raise Exception( 'Given state vector is not in state space. stateVector=' + str( state ) )
#        return int( indAr[0] )
#
#    def indexOfAction( self, action ):
#        '''
#        @return: Index of action in action space.
#        '''
#        return int( np.where( self.getActionSpace() == action )[0][0] )
#
#    def indexOfStateActionPair( self, sapair ):
#        '''
#        Get the index of a state action pair.
#
#        @param sapair: A pair (state, action)
#
#        @return: Index of the pair in state-action space (as returned by getStateActionPairIterable)
#        '''
#        indState = self.indexOfState( sapair[0] )
#        indAction = self.indexOfAction( sapair[1] )
#        return indAction * len( self.getStateSpace() ) + indState
#
#    def getStateActionPairIterable( self ):
#        '''
#        @return: An iterable object that iterates over all state action pairs.
#        '''
#
#        class SAIter:
#
#            def __init__( self, stateSpace, actionSpace ):
#                self.__stateSpace = stateSpace
#                self.__actionSpace = actionSpace
#                self.__ind = 0
#
#            def next( self ):
#                actInd = self.__ind / len( self.__stateSpace )
#                stateInd = self.__ind % len( self.__stateSpace )
#                self.__ind += 1
#                if self.__ind > len( self.__stateSpace ) * len( self.__actionSpace ):
#                    raise StopIteration
#                return ( self.__stateSpace[stateInd], self.__actionSpace[actInd] )
#
#        class SA:
#
#            def __init__( self, stateSpace, actionSpace ):
#                self.__stateSpace = stateSpace
#                self.__actionSpace = actionSpace
#
#            def __iter__( self ):
#                return SAIter( self.__stateSpace, self.__actionSpace )
#
#            def __len__( self ):
#                return len( self.__stateSpace ) * len( self.__actionSpace )
#
#            def __getitem__( self, i ):
#                return self.__getslice__( i, i + 1 )[0]
#
#            def __getslice__( self, i, j ):
#                res = []
#                for ind in range( i, j ):
#                    actInd = ind / len( self.__stateSpace )
#                    stateInd = ind % len( self.__stateSpace )
#                    res.append( ( self.__stateSpace[stateInd], self.__actionSpace[actInd] ) )
#                return res
#
#        return SA( self.getStateSpace(), self.getActionSpace() )

    def isGoalState( self, state ):
        return self.__isGoalState( state )

    def getStateSpace( self ):
        return self.__discretizedStateSpace

    def getActionSpace( self ):
        return self.__actionSpace

    def getGamma( self ):
        return self.__gamma





def computeTransitionModel( mdp, pi ):
    '''
    Compute the transition model for the given MDP with the given stochastic policy.
    
    @param mdp: TabularMDP instance.
    @param pi: Stochastic policy object.
    
    @return: Transition model as numpy matrix.
    '''
    Ppi = np.zeros( ( len( mdp.getStateSpace() ), len( mdp.getStateSpace() ) ) )
    piProb = pi.selectionProbabilityMatrix( mdp.getStateSpace() )
    for aind in range( len( mdp.getActionSpace() ) ):
        pia = np.diag( piProb[:, aind] )
        Pa = mdp.getTransitionMatrix( action=mdp.getActionSpace()[aind] )
        Ppi = Ppi + np.dot( pia, Pa )

    return Ppi

def stationaryDistribution( mdp, pi ):
    '''
    Stationary distribution of the MDP under the given stochastic policy.
    
    @param mdp: TabularMDP instance.
    @param pi: Stochastic policy instance.
    
    @return: A numpy array with the stationary distribution, each entry 
        corresponds to the probability of reaching a state.
    '''
    Ppi = computeTransitionModel( mdp, pi )

    sstartDistr = mdp.startStateDistribution()
    ds = np.array( sstartDistr, copy=True )
#    converged = False
#    t = 1
#    while not converged:
    for t in range( 1, 3001 ):
        dsUpdate = np.dot( np.linalg.matrix_power( Ppi, t ), sstartDistr )
#        dsUpdate /= np.sum( dsUpdate )
#        converged = np.linalg.norm( dsUpdate ) < 1.0e-15
        ds = ds + dsUpdate * ( mdp.getGamma() ** t )
#        t += 1
    ds *= 1.0 - mdp.getGamma()
#    ds /= np.sum( ds )

#    print 'ds mass: ' + str( np.sum( ds ) )

#    ds = np.ones( len( mdp.getStateSpace() ) )
    ds /= np.sum( ds )
    return ds

def stationaryDistributionActionValueFunction( mdp, pi, ds=None ):
    '''
    Compute the stationary distribution for a set of state-action pairs.
    @param mdp: Tabular MDP for which to calculate the transition model.
    @param pi: Stochastic policy generating the stationary distribution.
    @param stateTransitonModel: Transition model under the given policy from state
                to state, optional.
                
    @return: Numpy array with the stationary distribution over state-action pairs.
    '''
    if ds is None:
        ds = stationaryDistribution( mdp, pi )

    dsActionValueFunction = np.zeros( len( ds ) * len( mdp.getActionSpace() ) )

    piProb = pi.selectionProbabilityMatrix( mdp.getStateSpace() )
    for i in range( len( mdp.getActionSpace() ) ):
        iStart = i * len( mdp.getStateSpace() )
        dsActionValueFunction[iStart:iStart + len( mdp.getStateSpace() )] = ds * piProb[:, i]
    ds = dsActionValueFunction

    return ds

def computeTransitionModelStateAction( mdp, pi ):
    '''
    Compute a state-action pair transition model.
    
    @param mdp: Tabular MDP
    @param pi: Stochastic policy.
    
    @return: Transition model.
    '''
    Ppi = np.zeros( ( len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) , \
                      len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) ) )
    piProb = pi.selectionProbabilityMatrix( mdp.getStateSpace() )
    for a1 in range( len( mdp.getActionSpace() ) ):
        Ta1 = mdp.getTransitionMatrix( action=mdp.getActionSpace()[a1] )

        for a2 in range( len( mdp.getActionSpace() ) ):
            pia2 = piProb[:, a2]
#            Ta1a2 = np.dot( np.diag( pia2 ), Ta1 )
            Ta1a2 = np.dot( Ta1, np.diag( pia2 ) )

            low1 = a1 * len( mdp.getStateSpace() )
            high1 = ( a1 + 1 ) * len( mdp.getStateSpace() )
            low2 = a2 * len( mdp.getStateSpace() )
            high2 = ( a2 + 1 ) * len( mdp.getStateSpace() )

#            Ppi[low2:high2, low1:high1] = Ta1a2
            Ppi[low1:high1, low2:high2] = Ta1a2

    return Ppi

def computeMSPBEProjectionMatrix( stationaryDistribution, basisFunctionMatrix ):
    '''
    Calculate the MSPBE projection matrix for the given stationary distribution and basis function.
    
    @param stationaryDistribution: Stationary distribution
    @param basisFunctionMatrix: Basis function matrix.
    '''
    phiMat = basisFunctionMatrix

    try:
        D = np.array( np.diag( stationaryDistribution ) )
        invPDP = np.linalg.inv( np.dot( phiMat.T, np.dot( D, phiMat ) ) )
    except np.linalg.LinAlgError:
#        logging.warn( 'MSPBE calculation: Feature matrix is singular and inverse for projection matrix will be adjusted.' )
#        stationaryDistribution += 1.0e-10
#        stationaryDistribution /= np.sum( stationaryDistribution )
#        D = np.array( np.diag( stationaryDistribution ) )
        invPDP = np.linalg.pinv( np.dot( phiMat.T, np.dot( D, phiMat ) ) )

    Pi = np.dot( phiMat, np.dot( invPDP, np.dot( phiMat.T, D ) ) )
    return Pi

def mspbeStateActionValues( theta, mdp, basisFunction, stochasticPolicy, parametricPolicy=True, d_sa=None, d_s=None ):
    '''
    Compute the MSPBE for a state-action value function.
    
    @param theta: Value function parameter vector.
    @param mdp: MDP.
    @param basisFunction: State basis function.
    @param stochasticPolicy: Stochastic policy.
    
    @return: MSPBE for given theta.
    '''
    if parametricPolicy:
        pi = stochasticPolicy.copy()
        def q( s, a ):
            return np.dot( theta, basisFunction( s, a ) )
        pi.setActionValueFunction( q )
        stochasticPolicy = pi

    Ppi = computeTransitionModelStateAction( mdp, stochasticPolicy )
    # This reproduces the results from the Gradient Temporal-Difference Learning Algorithms thesis
#    Ppi = np.ones( ( 14, 14 ) ) / 14.
    if d_sa is None and d_s is None:
        d_sa = stationaryDistributionActionValueFunction( mdp, stochasticPolicy )
    elif d_sa is None:
        d_sa = stationaryDistributionActionValueFunction( mdp, stochasticPolicy, ds=d_s )
    R = []
    for s, a in mdp.getStateActionPairIterable():
        R.append( mdp.getRewardExpected( s, a ) )
    R = np.array( R )

    Phi = basisFunctionMatrixStateAction( basisFunction, mdp )
    Pi = computeMSPBEProjectionMatrix( d_sa, Phi )

    Q = np.dot( Phi, theta )
#    print 'Q values: ' + str( Q )
    err = np.dot( Pi, Q - ( R + mdp.getGamma() * np.dot( Ppi, Q ) ) )
#    err = Q - ( R + mdp.getGamma() * np.dot( Ppi, Q ) )
    mspbe = np.dot( err, d_sa * err )
    return mspbe

def mspbeStateValues( theta, mdp, basisFunction, stochasticPolicy, parametricPolicy=True, ds=None ):
    '''
    Compute the MSPBE for a state-action value function.
    
    @param theta: Value function parameter vector.
    @param mdp: MDP.
    @param basisFunction: State basis function.
    @param stochasticPolicy: Stochastic policy.
    
    @return: MSPBE for given theta.
    '''
    if parametricPolicy:
        pi = stochasticPolicy.copy()
        def q( s, a ):
            return np.dot( theta, basisFunction( s, a ) )
        pi.setActionValueFunction( q )
        stochasticPolicy = pi

    Ppi = computeTransitionModel( mdp, stochasticPolicy )
    if ds is None:
        ds = stationaryDistribution( mdp, stochasticPolicy )

    R = []
    for s in mdp.getStateSpace():
        r = 0
        for a in mdp.getActionSpace():
            r += stochasticPolicy.selectionProbability( s, a ) * mdp.getRewardExpected( s, a )
        R.append( r )
    R = np.array( R )

    Phi = basisFunctionMatrixState( basisFunction, mdp )
    Pi = computeMSPBEProjectionMatrix( ds, Phi )

    V = np.dot( Phi, theta )
    err = np.dot( Pi, V - ( R + mdp.getGamma() * np.dot( Ppi, V ) ) )
    mspbe = np.dot( err, ds * err )
    return mspbe

def msbeStateValues( theta, mdp, basisFunction, stochasticPolicy, parametricPolicy=True, errweights=None ):
    '''
    Compute the MSPBE for a state-action value function.
    
    @param theta: Value function parameter vector.
    @param mdp: MDP.
    @param basisFunction: State basis function.
    @param stochasticPolicy: Stochastic policy.
    
    @return: MSPBE for given theta.
    '''
    if parametricPolicy:
        pi = stochasticPolicy.copy()
        def q( s, a ):
            return np.dot( theta, basisFunction( s, a ) )
        pi.setActionValueFunction( q )
        stochasticPolicy = pi

    Ppi = computeTransitionModel( mdp, stochasticPolicy )
#    ds = stationaryDistribution( mdp, stochasticPolicy )

    R = []
    for s in mdp.getStateSpace():
        r = 0
        for a in mdp.getActionSpace():
            r += stochasticPolicy.selectionProbability( s, a ) * mdp.getRewardExpected( s, a )
        R.append( r )
    R = np.array( R )

    Phi = basisFunctionMatrixState( basisFunction, mdp )

    V = np.dot( Phi, theta )
#    print 'Q values: ' + str( Q )
    err = V - ( R + mdp.getGamma() * np.dot( Ppi, V ) )
#    err = Q - ( R + mdp.getGamma() * np.dot( Ppi, Q ) )

    msbe = np.linalg.norm( err )
    return msbe

def mstdeStateValues( theta, mdp, phi, pi ):
    '''
    Compute the expected value of the TD error squared, E[delta^2], where the expectation 
    is over the stationary state distribution and the next state distribution.
    
    @param theta: Parameter vector for action-values.
    @param mdp: MDP
    @param phi: Basis function (function of state pairs).
    @param pi: Stochastic behaviour policy.
    
    @return: MSTDE
    '''
    ds = stationaryDistribution( mdp, pi )
    err = 0
    for s in mdp.getStateSpace():
        si = mdp.indexOfState( s )
        prob = ds[si]

        tderror_s = 0

        for a in mdp.getActionSpace():
            nextStateDistr = mdp.getNextStateDistribution( s, a )
            for snext in mdp.getStateSpace():
                sj = mdp.indexOfState( snext )
                prob_transition = pi.selectionProbability( s, a ) * nextStateDistr[sj]
                if prob_transition == 0:
                    continue
                reward = mdp.getReward( s, a, snext )
                tderror_s = tderror_s + prob_transition * ( reward + mdp.getGamma() * np.dot( theta, phi( snext ) ) - np.dot( theta, phi( s ) ) ) ** 2

        err = err + prob * tderror_s

    return err

def mstdeStateActionValues( theta, mdp, phi, pi, parametricPolicy=True, d_sa=None, d_s=None ):
    '''
    Compute the expected value of the TD error squared, E[delta^2], where the expectation 
    is over the stationary state-action pair distribution and the next state-action pair
    distribution.
    
    @param theta: Parameter vector for action-values.
    @param mdp: MDP
    @param phi: Basis function (function of state-action pairs).
    @param pi: Stochastic behaviour policy.
    
    @return: MSTDE
    '''
    if parametricPolicy:
        pi = pi.copy()
        def q( s, a ):
            return np.dot( theta, phi( s, a ) )
        pi.setActionValueFunction( q )

    if d_sa is None and d_s is None:
        d_sa = stationaryDistributionActionValueFunction( mdp, pi )
    elif d_sa is None:
        d_sa = stationaryDistributionActionValueFunction( mdp, pi, ds=d_s )
#    ds = np.ones( len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) )
#    ds /= np.sum( ds )
    err = 0
    for s, a in mdp.getStateActionPairIterable():
        nextStateDistr = mdp.getNextStateDistribution( s, a )
        sai = mdp.indexOfStateActionPair( ( s, a ) )
        prob_sa = d_sa[sai]

        tderror_sa = 0

        for snext, anext in mdp.getStateActionPairIterable():
            sj = mdp.indexOfState( snext )
            prob_transition = nextStateDistr[sj] * pi.selectionProbability( snext, anext )

            if prob_transition == 0:
                continue
            reward = mdp.getReward( s, a, snext )
            tderror_sa = tderror_sa + prob_transition * ( reward + mdp.getGamma() * np.dot( theta, phi( snext, anext ) ) - np.dot( theta, phi( s, a ) ) ) ** 2

        err = err + prob_sa * tderror_sa

    return err



