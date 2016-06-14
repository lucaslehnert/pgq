'''
Created on Oct 7, 2015

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

This module contains implementations of different Q-learning algorithms.

'''

import numpy as np
from abc import abstractmethod

def argmaxrand( values ):
    indList = np.argwhere( values == np.max( values ) ).flatten()
    return np.random.choice( indList )

class LinearQLearning( object ):

    def __init__( self, basisFunction, initTheta, alpha=0.1 ):
        self._phi = basisFunction
        self._theta = np.array( initTheta, copy=True )
        self._alpha = alpha

    @abstractmethod
    def getUpdate( self, state, action, reward, stateNext ):
        raise NotImplementedError( 'Not implemented' )

    @abstractmethod
    def updateEstimates( self, estimateUpdate ):
        raise NotImplementedError( 'Not implemented' )

    def getTheta( self ):
        return np.array( self._theta, copy=True )

    @abstractmethod
    def getEstimateDim( self ):
        raise NotImplementedError( 'Not implemented' )

    @abstractmethod
    def getNextAction( self ):
        raise NotImplementedError( 'Not implemented' )

    def updateTransition( self, s, a, reward, snext ):
        update = self.getUpdate( s, a, reward, snext )
        self.updateEstimates( update )

    def getBasisFunction( self ):
        return self._phi

class SARSA( LinearQLearning ):
    '''
    Onpolicy SARSA(lambda) q-learning with replacing traces. For SARSA(0) set lambda to 0.
    The eligibility trace extension only works with binary vectors (e.g. tile coding).
    '''

    def __init__( self, gamma, behaviorPolicy, traceLambda=0.0, **params ):
        super( SARSA, self ).__init__( **params )
        self.__gamma = gamma
        self.__traceLambda = traceLambda
        self.__e = np.zeros( len( self._theta ) )
        self.__nextAction = None
        self._controlPolicy = behaviorPolicy

    def getUpdate( self, state, action, reward, stateNext ):
        def q( s, a ):
            return np.dot( self._theta, self._phi( s, a ) )
        self.__nextAction = self._controlPolicy.selectAction( stateNext, actionValueFunction=q )

        phi_t = self._phi( state, action )
        phi_tnext = self._phi( stateNext, self.__nextAction )
        delta_t = reward + self.__gamma * np.dot( self._theta, phi_tnext ) - np.dot( self._theta, phi_t )

        self.__e = self.__e + phi_t
        update = delta_t * self.__e
        self.__e = self.__gamma * self.__traceLambda * self.__e
        return update
#        self.__e = self.__gamma * self.__traceLambda * self.__e + phi_t
#        self.__e[self.__e > 1.0] = 1.0
#        return delta_t * self.__e

    def getNextAction( self ):
        return self.__nextAction

    def updateEstimates( self, thetaUpdate ):
        self._theta = self._theta + self._alpha * thetaUpdate

    def getEstimateDim( self ):
        return len( self._theta )


class GreedyQ( LinearQLearning ):

    def __init__( self, **params ):
        super( GreedyQ, self ).__init__( params['basisFunction'], params['initTheta'], params['alpha'] )

        self._e = np.zeros( len( self._theta ) )
        self._gamma = params['gamma']
        self._actionSpace = params['actionSpace']
        self._traceLambda = params.get( 'traceLambda', 0.0 )
        self._behaviorPolicy = params['behaviorPolicy']

    def getUpdate( self, s, a, reward, snext ):

        def q( s, a ):
            return np.dot( self._theta, self._phi( s, a ) )

        actionValues = map( lambda a : np.dot( self._theta, self._phi( s, a ) ), self._actionSpace )
        if np.dot( self._theta, self._phi( s, a ) ) == np.max( actionValues ):
            rho = 1.0 / self._behaviorPolicy.selectionProbability( s, a, actionValueFunction=q )
        else:
            rho = 0.0

        actionNextValues = map( lambda act : np.dot( self._theta, self._phi( snext, act ) ), self._actionSpace )
        actionNext = self._actionSpace[argmaxrand( actionNextValues )]
        phi_sanext = self._phi( snext, actionNext )
        phi_sa = self._phi( s, a )

        self._e = rho * phi_sa + self._gamma * self._traceLambda * self._e

        tderror = reward + self._gamma * np.dot( self._theta, phi_sanext ) - np.dot( self._theta, phi_sa )
        return tderror * self._e

    def updateEstimates( self, thetaUpdate ):
        self._theta = self._theta + self._alpha * thetaUpdate

    def getEstimateDim( self ):
        return len( self._theta )

class GreedyGQ( LinearQLearning ):

    def __init__( self, **params ):

        super( GreedyGQ, self ).__init__( params['basisFunction'], params['initTheta'], params['alpha'] )

        self._e = np.zeros( len( self._theta ) )
        self._w = np.zeros( len( self._theta ) )
        self._gamma = params['gamma']
        self._actionSpace = params['actionSpace']
        self._traceLambda = params.get( 'traceLambda', 0.0 )
        self._behaviorPolicy = params['behaviorPolicy']

        self._beta = params['beta']

    def getUpdate( self, s, a, reward, snext ):

        def q( s, a ):
            return np.dot( self._theta, self._phi( s, a ) )

        actionValues = map( lambda a : np.dot( self._theta, self._phi( s, a ) ), self._actionSpace )
        if np.dot( self._theta, self._phi( s, a ) ) == np.max( actionValues ):
            rho = 1.0 / self._behaviorPolicy.selectionProbability( s, a, actionValueFunction=q )
        else:
            rho = 0.0

        actionNextValues = map( lambda act : np.dot( self._theta, self._phi( snext, act ) ), self._actionSpace )
        actionNext = self._actionSpace[argmaxrand( actionNextValues )]
        phi_sanext = self._phi( snext, actionNext )
        phi_sa = self._phi( s, a )

        self._e = phi_sa + rho * self._gamma * self._traceLambda * self._e

        tderror = reward + self._gamma * np.dot( self._theta, phi_sanext ) - np.dot( self._theta, phi_sa )

        thetaUpdate = tderror * self._e - self._gamma * ( 1 - self._traceLambda ) * np.dot( phi_sa, self._w ) * phi_sanext
        wUpdate = tderror * self._e - np.dot( phi_sa, self._w ) * phi_sa
        return np.append( thetaUpdate, wUpdate )

    def updateEstimates( self, update ):
        self._theta = self._theta + self._alpha * update[:len( self._theta )]
        self._w = self._w + self._beta * update[len( self._theta ):]

    def getEstimateDim( self ):
        return len( self._theta ) + len( self._w )

class Q( LinearQLearning ):

    def __init__( self, **params ):
        super( Q, self ).__init__( params['basisFunction'], params['initTheta'], params['alpha'] )

        self._gamma = params['gamma']
        self._behaviorPolicy = params['behaviorPolicy']
        self._targetPolicy = params['targetPolicy']
        self._actionSpace = params['actionSpace']

    def getUpdate( self, s, a, reward, snext ):

        def q( s, a ):
            return np.dot( self._theta, self._phi( s, a ) )

        rho = self._targetPolicy.selectionProbability( s, a, actionValueFunction=q ) / self._behaviorPolicy.selectionProbability( s, a, actionValueFunction=q )

        phi_sanext = np.sum( map( lambda act: self._targetPolicy.selectionProbability( snext, act, actionValueFunction=q ) * self._phi( snext, act ), self._actionSpace ), axis=0 )
        phi_sa = self._phi( s, a )

        tderror = reward + self._gamma * np.dot( self._theta, phi_sanext ) - np.dot( self._theta, phi_sa )

        return rho * tderror * phi_sa

    def updateEstimates( self, thetaUpdate ):
        self._theta = self._theta + self._alpha * thetaUpdate

    def getEstimateDim( self ):
        return len( self._theta )

class GQ( LinearQLearning ):

    def __init__( self, **params ):
        super( GQ, self ).__init__( params['basisFunction'], params['initTheta'], params['alpha'] )

        self._w = np.zeros( len( self._theta ) )
        self._gamma = params['gamma']
        self._behaviorPolicy = params['behaviorPolicy']
        self._targetPolicy = params['targetPolicy']
        self._actionSpace = params['actionSpace']

        self._beta = params['beta']

    def getUpdate( self, s, a, reward, snext ):

        def q( s, a ):
            return np.dot( self._theta, self._phi( s, a ) )

        rho = self._targetPolicy.selectionProbability( s, a, actionValueFunction=q ) / self._behaviorPolicy.selectionProbability( s, a, actionValueFunction=q )

        phi_sanext = np.sum( map( lambda act: self._targetPolicy.selectionProbability( snext, act, actionValueFunction=q ) * self._phi( snext, act ), self._actionSpace ), axis=0 )
        phi_sa = self._phi( s, a )

        tderror = reward + self._gamma * np.dot( self._theta, phi_sanext ) - np.dot( self._theta, phi_sa )

        thetaUpdate = rho * ( tderror * phi_sa - self._gamma * np.dot( phi_sa, self._w ) * phi_sanext )
        wUpdate = rho * ( tderror * phi_sa - np.dot( phi_sa, self._w ) * phi_sa )
        return np.append( thetaUpdate, wUpdate )

    def updateEstimates( self, update ):
        self._theta = self._theta + self._alpha * update[:len( self._theta )]
        self._w = self._w + self._beta * update[len( self._theta ):]

    def getEstimateDim( self ):
        return len( self._theta ) + len( self._w )

class PGQ( LinearQLearning ):

    def __init__( self, **params ):
        super( PGQ, self ).__init__( params['basisFunction'], params['initTheta'], params['alpha'] )

        self._w = np.zeros( len( self._theta ) )
        self._gamma = params['gamma']
        self._behaviorPolicy = params['behaviorPolicy']
        self._targetPolicy = params['targetPolicy']
        self._actionSpace = params['actionSpace']

        self._beta = params['beta']

    def getUpdate( self, s, a, reward, snext ):

        # getParameterGradient( self, state, action, basisFunctionStateAction, actionValueFunction=None ):

        def q( s, a ):
            return np.dot( self._theta, self._phi( s, a ) )

        rho = self._targetPolicy.selectionProbability( s, a, actionValueFunction=q ) \
                / self._behaviorPolicy.selectionProbability( s, a, actionValueFunction=q )
        rhoGrad = self._targetPolicy.getParameterGradient( s, a, self._phi, actionValueFunction=q ) \
                / self._behaviorPolicy.selectionProbability( s, a, actionValueFunction=q )

        phi_sanext = np.sum( map( \
                        lambda act: self._targetPolicy.selectionProbability( snext, act, actionValueFunction=q ) * self._phi( snext, act ), \
                        self._actionSpace ), axis=0 )
        phi_sa = self._phi( s, a )

        v_grad = np.sum( map( lambda act : self._targetPolicy.getParameterGradient( snext, act, self._phi, actionValueFunction=q ) \
                                            * np.dot( self._theta, self._phi( snext, act ) ) , self._actionSpace ) , axis=0 )

        tderror = reward + self._gamma * np.dot( self._theta, phi_sanext ) - np.dot( self._theta, phi_sa )

        thetaUpdate = rho * tderror * phi_sa - self._gamma * rho * np.dot( phi_sa, self._w ) * phi_sanext \
                    - rhoGrad * tderror * np.dot( phi_sa, self._w ) - self._gamma * rho * v_grad * np.dot( phi_sa, self._w ) \
                    + 1. / 2. * rhoGrad * np.dot( self._w, phi_sa ) ** 2

#        print 's,a,r,s\': ' + str( s ) + ',' + str( a ) + ',' + str( reward ) + ',' + str( snext )
#        print '\tpi grad= ' + str( self._targetPolicy.getParameterGradient( s, a, self._phi, actionValueFunction=q ) )
#        print '\tpi=      ' + str( self._behaviorPolicy.selectionProbability( s, a, actionValueFunction=q ) )
#        print '\trho=     ' + str( rho )
#        print '\trhoGrad= ' + str( rhoGrad )
#        print '\tphi\'=   ' + str( phi_sanext )
#        print '\tv_grad=  ' + str( v_grad )
#        print '\ttheta=   ' + str( self._theta )

        wUpdate = rho * ( tderror * phi_sa - np.dot( phi_sa, self._w ) * phi_sa )
        return np.append( thetaUpdate, wUpdate )

    def updateEstimates( self, update ):
        self._theta = self._theta + self._alpha * update[:len( self._theta )]
        self._w = self._w + self._beta * update[len( self._theta ):]

    def getEstimateDim( self ):
        return len( self._theta ) + len( self._w )

