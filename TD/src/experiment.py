'''
Created on Oct 7, 2015

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''
import numpy as np
from mdp import mspbeStateActionValues, mstdeStateActionValues

def experimentSampleTransitions( iterations, mdp, pi, agent, errorMeasures=[] ):
    '''
    Sample transitions using the state distribution of the MDP, its transition model and the current policy (the 
    action value function is computed using the agent).
    
    @param iterations: Maximum length of a trajectory.
    @param mdp: MDP used to simulate trajectories.
    @param pi: Policy used to select actions. The action-value function is computed using the agetn.getTheta() and 
            agent.getBasisFunction() methods.
    @param agent: Control agent used for an experiment, must be an instance of qlearning.LinearQLearning
    @param errorMeasures: List of error measures computed during the experiment. This is passed as a list of function pointers 
            with the agent being the only argument.
    
    @return: errorBenchmarks, completed
            errorBenchmarks containa a list of return values for the errorMeasure. completed is a boolean indicating if the 
            trajectory was completed or if a nan-error or value error was detected (indicating divergence).
    
    '''

    errorBenchmarks = []
    for err in errorMeasures:
        errorBenchmarks.append( [err( agent )] )

    completed = False
    try:
        for _ in range( iterations ):
            def q( s, a ):
                return np.dot( agent.getTheta(), agent.getBasisFunction()( s, a ) )

            s = mdp.sampleStartState()
            a = pi.selectAction( s, q )
            snext = mdp.sampleNextState( s, a )

            reward = mdp.getReward( s, a, snext )

            agent.updateTransition( s, a, reward, snext )

            for i in range( len( errorBenchmarks ) ):
                errorBenchmarks[i].append( errorMeasures[i]( agent ) )
        completed = True
    except Exception as e:
        print e
        completed = False

    return errorBenchmarks, completed

def experimentSimulateTransitions( iterations, mdp, pi, agent, errorMeasures=[], transitionListener=[], actionFromAgent=False ):
    '''
    Simulate transitions through an MDP with the given policy and agent.
    
    @param iterations: Maximum length of a trajectory.
    @param mdp: MDP used to simulate trajectories.
    @param pi: Policy used to select actions. The action-value function is computed using the agetn.getTheta() and 
            agent.getBasisFunction() methods.
    @param agent: Control agent used for an experiment, must be an instance of qlearning.LinearQLearning
    @param errorMeasures: List of error measures computed during the experiment. This is passed as a list of function pointers 
            with the agent being the only argument.
    @param transitionListener: List of function pointers listening to MDP transitions. The function signature is fn(s,a,r,sn).
    @param actionFromAgent: Boolean, default is false. If true, obtain the next action by calling agent.getNextAction().
    
    @return: errorBenchmarks, transitionBenchmark, completed
            errorBenchmarks and transitionBenchmark contain each a list of return values for the errorMeasure and
            transitionListener. completed is a boolean indicating if the trajectory was completed or if a nan-error
            or value error was detected (indicating divergence).
    
    '''

    errorBenchmarks = []
    for err in errorMeasures:
        errorBenchmarks.append( [err( agent )] )
    transitionBenchmark = []
    for _ in transitionListener:
        transitionBenchmark.append( [] )

    completed = False
    try:
        s = mdp.sampleStartState()
        for i in range( iterations ):
            if actionFromAgent and i > 0:
                a = agent.getNextAction()
            else:
                def q( s, a ):
                    return np.dot( agent.getTheta(), agent.getBasisFunction()( s, a ) )
                a = pi.selectAction( s, q )

            snext = mdp.sampleNextState( s, a )

            reward = mdp.getReward( s, a, snext )

            agent.updateTransition( s, a, reward, snext )

            for i in range( len( errorBenchmarks ) ):
                errorBenchmarks[i].append( errorMeasures[i]( agent ) )
            for i in range( len( transitionBenchmark ) ):
                transitionBenchmark[i].append( transitionListener[i]( s, a, snext, reward ) )

            s = snext
            if mdp.isGoalState( s ):
                break

        completed = True
    except Exception as e:
        print e
        completed = False


    return errorBenchmarks, transitionBenchmark, completed


def experimentDynamicProgrammingSweeps( sweeps, mdp, pi, agent, errorMeasures=[], d_s=None ):
    '''
    @deprecated: Delete this.
    '''
    if d_s is None:
        d_s = np.ones( len( mdp.getStateSpace() ) ) / float( len( mdp.getStateSpace() ) )


    errorBenchmarks = []
    for err in errorMeasures:
        errorBenchmarks.append( [err( agent )] )

    for _ in range( sweeps ):
        def q( s, a ):
            return np.dot( agent.getTheta(), agent.getBasisFunction()( s, a ) )

        est = np.zeros( agent.getEstimateDim() )
        for s, a in mdp.getStateActionPairIterable():
            ind_s = mdp.indexOfState( s )
            nextStateDistr = mdp.getNextStateDistribution( s, a )
            for snext in mdp.getStateSpace():
                ind_snext = mdp.indexOfState( snext )
                prob = d_s[ind_s] * pi.selectionProbability( s, a, q ) * nextStateDistr[ind_snext]
                if prob == 0.0:
                    continue

                reward = mdp.getReward( s, a, snext )
                est += prob * agent.getUpdate( s, a, reward, snext )

        agent.updateEstimates( est )

        for i in range( len( errorBenchmarks ) ):
            try:
                errorBenchmarks[i].append( errorMeasures[i]( agent ) )
            except:
                errorBenchmarks[i].append( np.NaN )

    return errorBenchmarks


def evaluateQlearningDP( numberOfSweeps, agent, mdp, phi, piBehavior, piTarget, stationary_distribution=None ):
    '''
    @deprecated: Delete this.
    '''
    if stationary_distribution is None:
        stationary_distribution = np.ones( len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) )

    mspbe = lambda t : mspbeStateActionValues( t, mdp, phi, piTarget, parametricPolicy=True, d_sa=stationary_distribution )
    mspbeList = [ mspbe( agent.getTheta() ) ]
    mstde = lambda t : mstdeStateActionValues( t, mdp, phi, piTarget, parametricPolicy=True, d_sa=stationary_distribution )
    mstdeList = [ mstde( agent.getTheta() ) ]
    thetaList = [ agent.getTheta() ]

#    dsa = np.ones( len( mdp.getStateSpace() ) * len( mdp.getActionSpace() ) )
#    dsa /= np.sum( dsa )
    dsa = stationary_distribution

    for _ in range( numberOfSweeps ):

        try:
            for s, a in mdp.getStateActionPairIterable():
                sai = mdp.indexOfStateActionPair( ( s, a ) )
                prob = dsa[sai]
                def q( s, a ):
                    return np.dot( agent.getTheta(), agent.phi( s, a ) )

                grad_theta = np.zeros( agent.getEstimateDim() )

                nextStateDistr = mdp.getNextStateDistribution( s, a )
                for snext in mdp.getStateSpace():
                    sj = mdp.indexOfState( snext )
                    prob_transition = nextStateDistr[sj]
                    if prob_transition == 0.0:
                        continue

                    reward = mdp.getReward( s, a, snext )
                    grad_theta = grad_theta + prob_transition * agent.getUpdate( s, a, reward, snext )

                agent.updateEstimates( prob * grad_theta )
#                print 's,a: ' + str( s ) + ',' + str( a ) + ', theta=' + str( agent.getTheta() )
        except ValueError, e:
            print e
            continue

        mspbeList.append( mspbe( agent.getTheta() ) )
        mstdeList.append( mstde( agent.getTheta() ) )
        thetaList.append( agent.getTheta() )

    return mspbeList, mstdeList, thetaList

def evaluateQlearningSimulated( numberOfEpisodes, agent, mdp, phi, piBehavior, piTarget, max_episode_length=10000 ):
    '''
    @deprecated: Delete this.
    '''
    mspbe = lambda t : mspbeStateActionValues( t, mdp, phi, piTarget, parametricPolicy=True )
    mstde = lambda t : mstdeStateActionValues( t, mdp, phi, piTarget, parametricPolicy=True )

    mspbeListEp = [ mspbe( agent.getTheta() ) ]
    mstdeListEp = [ mstde( agent.getTheta() ) ]
    thetaListEp = [agent.getTheta()]
    episodeLength = []

    for epInd in range( numberOfEpisodes ):
        print 'Running episode ' + str( epInd )
        try:
            s = mdp.sampleStartState()

            t = 0
            while not mdp.isGoalState( s ):
                def q( s, a ):
                    return np.dot( agent.getTheta(), agent.getBasisFunction()( s, a ) )
                a = piBehavior.selectAction( s, actionValueFunction=q )
                snext = mdp.sampleNextState( s, a )
                reward = mdp.getReward( s, a, snext )

                agent.updateTransition( s, a, reward, snext )

                s = snext
                t += 1
                if t >= max_episode_length:
                    break
            print 'Episode length: ' + str( t + 1 )

            mspbeListEp.append( mspbe( agent.getTheta() ) )
            mstdeListEp.append( mstde( agent.getTheta() ) )
            thetaListEp.append( agent.getTheta() )
            episodeLength.append( t + 1 )


        except ValueError, e:
            print e
            continue

    resultDict = { 'episodeLength' : episodeLength, 'mspbeEp' : mspbeListEp, 'mstdeEp' : mstdeListEp, 'thetaEp' : thetaListEp }

    return resultDict

