'''
Created on Sep 18, 2015

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

This module contains different policy implementations.

'''
import numpy as np
import random
import scipy.stats

from abc import abstractmethod

class Policy( object ):
    '''
    This class must be extended by any policy in the rl package.
    '''

    def __init__( self, name='Policy' ):
        '''
        Construct a policy with the given name.
        
        @param name: Policy name.
        '''
        self.__name = name

    def getName( self ):
        return self.__name

    @abstractmethod
    def selectAction( self, state, actionValueFunction=None ):
        '''
        Select an action at the given state using this policy.
        
        @param state: State at which to choose an action.
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: The action that should be selected.
        '''
        return

    @abstractmethod
    def selectionProbability( self, state, action, actionValueFunction=None ):
        '''
        Get the action selection probability at the given state.
        
        @param state: State
        @param action: Action
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: Probability of selecting the given action at the given state.
        '''
        return

    def selectionProbabilityMatrix( self, stateSet, actionValueFunction=None ):
        '''
        @return: Action selection probability matrix. Each column corresponds to a 
            specific action and contains the selection probabilities with the same
            ordering as the givens state set.        
        '''
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct
        Pa = np.zeros( ( len( stateSet ), len( self.__actionSpace ) ) )
        for i in range( len( stateSet ) ):
            for j in range( len( self.__actionSpace ) ):
                Pa[i, j] = self.selectionProbability( stateSet[i], self.__actionSpace[j], actionValueFunction=actionValueFunction )

        return Pa

    def setActionValueFunction( self, q ):
        '''
        Overwrites the action-value function for the given policy.
        '''
        self.__actionValueFnct = q


    @abstractmethod
    def copy( self ):
        '''
        @return: A copy of the policy object.
        '''

class GreedyPolicy( Policy ):
    '''
    This policy selects actions greedily with respect to an action value function.
    If multiple actions tie with the best action value, an action is chosen uniformly
    at random.

    The action selection probability is 1.0 if the given action has an optimal action
    value, otherwise it is 0.0.
    '''

    def __init__( self, actionSpace, defaultActionValueFunction=None ):
        '''
        Constructs a deterministic policy. This policy selects an action that is
        pi with respect to the given action value function.

        @param actionSpace: Action space from which to select actions.
        @param defaultActionValueFunction: Action-value function function that is
            used to select the action with the highest value. If none is specifed
            a constant zero value function is assumed.
        '''
        super( GreedyPolicy, self ).__init__( name='GreedyPolicy' )

        if defaultActionValueFunction == None:
            def q( s, a ):
                return 0
            self.__actionValueFnct = q
        else:
            self.__actionValueFnct = defaultActionValueFunction

        self.__actionSpace = actionSpace

    def selectAction( self, state, actionValueFunction=None ):
        '''
        Select an action at the given state using this policy.

        @param state: State at which to choose an action.
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: The action that should be selected.
        '''
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct

        qAr = np.array( [actionValueFunction( state, a ) for a in self.__actionSpace ] )
        if not np.all( np.isfinite( qAr ) ):
            raise ValueError( 'Action values contain not-a-number values or are not finite. qAr=' + str( qAr ) )

        bestActionIndList = np.where( qAr == np.max( qAr ) )[0]
        bestActionInd = bestActionIndList[random.randrange( len( bestActionIndList ) )]
        return self.__actionSpace[bestActionInd]

    def selectionProbability( self, state, action, actionValueFunction=None ):
        '''
        Get the action selection probability at the given state.

        @param state: State
        @param action: Action
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: Probability of selecting the given action at the given state.
        '''
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct

        qAr = np.array( [actionValueFunction( state, a ) for a in self.__actionSpace ] )
        if not np.all( np.isfinite( qAr ) ):
            raise ValueError( 'Action values contain not-a-number values or are not finite. qAr=' + str( qAr ) )

#        bestActionInd = np.where( qAr == np.max( qAr ) )[0][0]
#        if self.__actionSpace[bestActionInd] == action:
#            return 1.0
#        else:
#            return 0
#
        actionInd = np.where( self.__actionSpace == action )[0][0]
        if qAr[actionInd] == np.max( qAr ):
            return 1.0 / float( len( np.where( qAr == np.max( qAr ) )[0] ) )
#            return 1.0
        else:
            return 0.0



    def copy( self ):
        return GreedyPolicy( self.__actionSpace, defaultActionValueFunction=self.__actionValueFnct )

    def selectionProbabilityMatrix( self, stateSet, actionValueFunction=None ):
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct

        Pa = np.zeros( ( len( stateSet ), len( self.__actionSpace ) ) )
        for i in range( len( stateSet ) ):
            qAr = np.array( [actionValueFunction( stateSet[i], a ) for a in self.__actionSpace ] )
            if not np.all( np.isfinite( qAr ) ):
                raise ValueError( 'Action values contain not-a-number values or are not finite. qAr=' + str( qAr ) )

#            nextActInd = np.random.choice( np.where( qAr == qAr.max() )[0] )
##            nextActInd = np.argmax( qAr )
#            Pa[i, nextActInd] = 1.0
#            Pa[i] *= 0.0
            for j in range( len( self.__actionSpace ) ):
                if qAr[j] == np.max( qAr ):
                    Pa[i, j] = 1.0 / float( len( np.where( qAr == np.max( qAr ) )[0] ) )
#                    Pa[i, j] = 1.0
#                    break
                else:
                    Pa[i, j] = 0.0


        return Pa

    def setActionValueFunction( self, q ):
        '''
        Overwrites the action-value function for the given policy.
        '''
        self.__actionValueFnct = q

class EGreedyPolicy( Policy ):
    '''
    This class implements the E-pi policy. It selects an action randomly with epsilon
    probability uniformly at random. With 1-epsilon an action is chosen greedily with respect
    to the action-value function.
    '''

    def __init__( self, actionSpace, epsilon=0.1, defaultActionValueFunction=None ):
        '''
        Constructs an e-pi policy. This policy selects an action randomly with epsilon
        probability uniformly at random. With 1-epsilon an action is chosen greedily with 
        respect to the action-value function.
        
        @param actionSpace: Action space from which to select actions.
        @param epsilon: Epsilon parameter, a float or an object with a value() method, default is 0.1
        @param defaultActionValueFunction: Action-value function function that is
            used to select the action with the highest value. If none is specifed
            a constant zero value function is assumed.
        '''
        super( EGreedyPolicy, self ).__init__( name='EGreedyPolicy_epsilon_' + str( epsilon ) )

        if defaultActionValueFunction == None:
            def q( s, a ):
                return 0
            self.__actionValueFnct = q
        else:
            self.__actionValueFnct = defaultActionValueFunction

        self.__actionSpace = actionSpace
        self.__epsilon = epsilon

    def selectAction( self, state, actionValueFunction=None ):
        '''
        Select an action at the given state using this policy.
        
        @param state: State at which to choose an action.
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: The action that should be selected.
        '''
        if hasattr( self.__epsilon, 'value' ) and callable( getattr( self.__epsilon, 'value' ) ):
            epsilon = self.__epsilon.value()
        else:
            epsilon = self.__epsilon

        if scipy.stats.bernoulli.rvs( epsilon ) == 1:
            actionInd = random.randrange( len( self.__actionSpace ) )
        else:
            if actionValueFunction == None:
                actionValueFunction = self.__actionValueFnct

            qAr = [actionValueFunction( state, a ) for a in self.__actionSpace]
            if not np.all( np.isfinite( qAr ) ):
                raise ValueError( 'Action values contain not-a-number values or are not finite. qAr=' + str( qAr ) )
            bestActionIndList = np.where( qAr == np.max( qAr ) )[0]
            actionInd = bestActionIndList[random.randrange( len( bestActionIndList ) )]
        return self.__actionSpace[actionInd]

    def selectionProbability( self, state, action, actionValueFunction=None ):
        '''
        Get the action selection probability at the given state.
        
        @param state: State
        @param action: Action
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: Probability of selecting the given action at the given state.
        '''
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct

        if hasattr( self.__epsilon, 'value' ) and callable( getattr( self.__epsilon, 'value' ) ):
            epsilon = self.__epsilon.value()
        else:
            epsilon = self.__epsilon

        qAr = [actionValueFunction( state, a ) for a in self.__actionSpace]
        if not np.all( np.isfinite( qAr ) ):
            raise ValueError( 'Action values contain not-a-number values or are not finite. qAr=' + str( qAr ) )
        actionInd = np.where( self.__actionSpace == action )[0][0]
        if qAr[actionInd] == np.max( qAr ):
            prob = 1.0 / float( len( self.__actionSpace ) ) * epsilon \
                 + 1.0 / float( len( np.where( qAr == np.max( qAr ) )[0] ) ) * ( 1 - epsilon )
        else:
            prob = 1.0 / float( len( self.__actionSpace ) ) * epsilon + 0.0
        return prob

    def copy( self ):
        return EGreedyPolicy( self.__actionSpace, epsilon=self.__epsilon, defaultActionValueFunction=self.__actionValueFnct )

    def selectionProbabilityMatrix( self, stateSet, actionValueFunction=None ):
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct

        if hasattr( self.__epsilon, 'value' ) and callable( getattr( self.__epsilon, 'value' ) ):
            epsilon = self.__epsilon.value()
        else:
            epsilon = self.__epsilon

        Pa = np.zeros( ( len( stateSet ), len( self.__actionSpace ) ) )
        for i in range( len( stateSet ) ):
            qAr = np.array( [actionValueFunction( stateSet[i], a ) for a in self.__actionSpace ] )
            if not np.all( np.isfinite( qAr ) ):
                raise ValueError( 'Action values contain not-a-number values or are not finite. qAr=' + str( qAr ) )

            for j in range( len( self.__actionSpace ) ):
                if qAr[j] == np.max( qAr ):
                    Pa[i, j] = 1.0 / float( len( self.__actionSpace ) ) * epsilon \
                             + 1.0 / float( len( np.where( qAr == np.max( qAr ) )[0] ) ) * ( 1 - epsilon )
                else:
                    Pa[i, j] = 1.0 / float( len( self.__actionSpace ) ) * epsilon + 0.0

        return Pa

    def setActionValueFunction( self, q ):
        '''
        Overwrites the action-value function for the given policy.
        '''
        self.__actionValueFnct = q

class BoltzmannPolicy( Policy ):
    '''
    Boltzmann exploration policy which selects actions according to the Boltzmann distribution
    using an action value function and a temperature parameter.
    '''

    def __init__( self, actionSpace, temperature=1.0, defaultActionValueFunction=None ):
        '''
        Constructs a Boltzmann exploration policy. This policy selects selects actions 
        according to the Boltzmann distribution using an action value function and a 
        temperature parameter.
        
        @param actionSpace: Action space from which to select actions.
        @param temperature: Temperature parameter, a float or an object with a value() method, default is 1.0
        @param defaultActionValueFunction: Action-value function function that is
            used to select the action with the highest value. If none is specifed
        '''
        super( BoltzmannPolicy, self ).__init__( name='BoltzmannPolicy_temp_' + str( temperature ) )

        if defaultActionValueFunction == None:
            def q( s, a ):
                return 0
            self.__actionValueFnct = q
        else:
            self.__actionValueFnct = defaultActionValueFunction

        self.__actionSpace = actionSpace
        self.__temperature = temperature

    def getTemperature( self ):
        '''
        @return: Temperature parameter of the policy.
        '''
        if hasattr( self.__temperature, 'value' ) and callable( getattr( self.__temperature, 'value' ) ):
            temp = self.__temperature.value()
        else:
            temp = self.__temperature
        return temp

    def selectAction( self, state, actionValueFunction=None ):
        '''
        Select an action at the given state using this policy.
        
        @param state: State at which to choose an action.
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: The action that should be selected.
        '''
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct

        if hasattr( self.__temperature, 'value' ) and callable( getattr( self.__temperature, 'value' ) ):
            temp = self.__temperature.value()
        else:
            temp = self.__temperature

        qAr = np.array( [actionValueFunction( state, a ) for a in self.__actionSpace ] )
        qAr = np.exp( qAr / temp )
        qAr /= np.sum( qAr )
        if not np.all( np.isfinite( qAr ) ):
            raise ValueError( 'Action values contain not-a-number values or are not finite. qAr=' + str( qAr ) )

        rv = scipy.stats.rv_discrete( name='action_selection_distr', values=( range( len( self.__actionSpace ) ), qAr ) )
        actionInd = rv.rvs( size=1 )[0]
        return self.__actionSpace[actionInd]

    def selectionProbability( self, state, action=None, actionValueFunction=None ):
        '''
        Get the action selection probability at the given state.
        
        @param state: State
        @param action: Action
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: Probability of selecting the given action at the given state.
        '''
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct

        if hasattr( self.__temperature, 'value' ) and callable( getattr( self.__temperature, 'value' ) ):
            temp = self.__temperature.value()
        else:
            temp = self.__temperature

        qAr = np.array( [actionValueFunction( state, a ) for a in self.__actionSpace ] )
        qAr = np.exp( qAr / temp )
        qAr /= np.sum( qAr )
        if not np.all( np.isfinite( qAr ) ):
            raise ValueError( 'Action values contain not-a-number values or are not finite. qAr=' + str( qAr ) )

        if action != None:
            actionInd = np.where( self.__actionSpace == action )[0][0]
            return qAr[actionInd]
        else:
            return qAr


    def copy( self ):
        return BoltzmannPolicy( self.__actionSpace, temperature=self.__temperature, defaultActionValueFunction=self.__actionValueFnct )

    def selectionProbabilityMatrix( self, stateSet, actionValueFunction=None ):
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct

        Pa = map( lambda s : self.selectionProbability( s, actionValueFunction=actionValueFunction ), stateSet )
        Pa = np.array( Pa )
        return Pa

    def getParameterGradient( self, state, action, basisFunctionStateAction, actionValueFunction=None ):
        '''
        @return: The gradient of the policy with respect to the action value parameter theta. The gradient is evaluated 
            for the given (state,action) pair.
        '''
        if actionValueFunction == None:
            actionValueFunction = self.__actionValueFnct
#        phiDim = len( basisFunctionStateAction( state, action ) )
#        grad = np.zeros( phiDim )
#        for i in range( phiDim ):
##            phiBar = np.sum( map( lambda b: basisFunctionStateAction( state, b )[i], self.__actionSpace ) )
#            phiBar = np.sum( map( lambda b: basisFunctionStateAction( state, b )[i] * self.selectionProbability( state, b, q ), self.__actionSpace ) )
#            grad[i] = 1.0 / self.getTemperature() * self.selectionProbability( state, action, q ) \
#                                    * ( basisFunctionStateAction( state, action )[i] - phiBar )
#        return grad

        piProb = self.selectionProbability( state, actionValueFunction=actionValueFunction )
        pi_sa = piProb[np.where( self.__actionSpace == action )[0][0]]

        phi_s = np.array( map( lambda a : basisFunctionStateAction( state, a ), self.__actionSpace ) )
        pi_phi = np.zeros( len( phi_s[0] ) )
        for i in range( len( piProb ) ):
            pi_phi += piProb[i] * phi_s[i]

        grad = 1.0 / self.getTemperature() * pi_sa * ( basisFunctionStateAction( state, action ) - pi_phi )
        return grad

    def setActionValueFunction( self, q ):
        '''
        Overwrites the action-value function for the given policy.
        '''
        self.__actionValueFnct = q



class UniformPolicy( object ):
    '''
    This class must be extended by any policy in the rl package. Note that
    setting an action value function does not have an effect.
    '''

    def __init__( self, actionSpace, name='UniformPolicy' ):
        '''
        Construct a policy with the given name.
        
        @param name: Policy name.
        '''
        super( UniformPolicy, self ).__init__( name='UniformPolicy' )
#        self.__name = name
        self.__actionSpace = actionSpace
        def q( s, a ):
            return 1.0
        self.__actionValueFnct = q

    def getName( self ):
        return self.__name

    @abstractmethod
    def selectAction( self, state, actionValueFunction=None ):
        '''
        Select an action at the given state using this policy.
        
        @param state: State at which to choose an action.
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: The action that should be selected.
        '''
        from random import randint
        actionInd = randint( 0, len( self.__actionSpace ) - 1 )
        return self.__actionSpace[actionInd]

    def selectionProbability( self, state, action, actionValueFunction=None ):
        '''
        Get the action selection probability at the given state.
        
        @param state: State
        @param action: Action
        @param actionValueFunction: Action value function that should be used,
            this is optional.
        @return: Probability of selecting the given action at the given state.
        '''
        return 1.0 / float( len( self.__actionSpace ) )

    def selectionProbabilityMatrix( self, stateSet, actionValueFunction=None ):
        Pa = np.ones( ( len( stateSet ), len( self.__actionSpace ) ) ) * 1.0 / float( len( self.__actionSpace ) )
        return Pa

    def copy( self ):
        return UniformPolicy( self.__actionSpace )

    def setActionValueFunction( self, q ):
        '''
        Overwrites the action-value function for the given policy.
        '''
        self.__actionValueFnct = q

class TabularProbabilityPolicy( Policy ):
    '''
    A policy that selects actions with a fixed probability. The selection probabilities 
    are the same across the state space.
    '''

    def __init__( self, selectionProb, actionSpace ):
        super( TabularProbabilityPolicy, self ).__init__( name='TabularProbabilityPolicy' )
        self.__selectionProb = selectionProb
        self.__actionSpace = actionSpace

    def selectAction( self, state, actionValueFunction=None ):
        rv = scipy.stats.rv_discrete( name='action_selection_distr', values=( range( len( self.__actionSpace ) ), self.__selectionProb ) )
        actionInd = rv.rvs( size=1 )[0]
        return self.__actionSpace[actionInd]

    def selectionProbability( self, state, action, actionValueFunction=None ):
        actionInd = np.where( self.__actionSpace == action )[0][0]
        return self.__selectionProb[actionInd]

    def selectionProbabilityMatrix( self, stateSet, actionValueFunction=None ):
        Pa = np.array( map( lambda s: self.__selectionProb, stateSet ) )
        return Pa

    def copy( self ):
        return TabularProbabilityPolicy( self.__selectionProb, self.__actionSpace )
