'''
Created on May 21, 2015

This module contains different basis function implementations.

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''

import numpy as np

def polynomialBasisFunctionSingleVariable( degree ):
    '''
    @deprecated: Use get1DPolynomialBasisFunction.
    Construct a single variable polynomial basis function.
    
    @param degree: Degree of the polynomial.
    
    @return: Basis function, a function of a single floating point variable.
    '''
    exponents = np.array( range( 0, degree + 1 ) )
    def basisFunction( x ):
        return np.power( x, exponents )
    return basisFunction

def tabularStateBasisFunction( tabularMDP=None, stateSpace=None ):
    '''
    @deprecated: Use get1DLabelSetBinaryBasisFunction.
    Create a basis function for the MDP. Can be called in the following ways:
        
        tabularStateBasisFunction(tabularMDP)
        tabularStateBasisFunction(stateSpace=S)
    
    @param tabularMDP: Tabular MDP.
    @param stateSpace: State space.
    
    @return: phi, dim. Phi is a function handle mapping a state to a numpy 
        array (vector). dim is the dimension of the returned array.
    '''
    if stateSpace == None:
        stateSpace = tabularMDP.getStateSpace()
    def phi( s ):
        sInd = np.where( np.all( stateSpace == s, axis=1 ) )[0][0]
        p = np.zeros( len( stateSpace ) )
        p[sInd] = 1.0
        return p
    return phi, len( stateSpace )

def tiling2DStateBasisFunction( tabularMDP=None, stateSpace=None, tiling=1 ):
    '''
    @deprecated: Use tiledStateBinaryBasisFunction.
    Compute a tiled basis function. an be called in the following ways:
        
        tabularStateBasisFunction(tabularMDP)
        tabularStateBasisFunction(stateSpace=S)
        tabularStateBasisFunction(tabularMDP, tiling=1)
        tabularStateBasisFunction(stateSpace=S, tiling=1)
    
    @param tabularMDP: Tabular MDP.
    @param stateSpace: State space.
    @param tiling: Tile size, default 1
    
    @return: phi, dim. Phi is a function handle mapping a state to a numpy 
        array (vector). dim is the dimension of the returned array.
    '''
    if tiling == 1:
        return tabularStateBasisFunction( tabularMDP, stateSpace )

    if stateSpace == None:
        stateSpace = tabularMDP.getStateSpace()

    if hasattr( tiling, '__iter__' ):
        xTiling = tiling[0]
        yTiling = tiling[1]
    else:
        xTiling = tiling
        yTiling = tiling

    indexMap = {}
    xmin = int( np.min( stateSpace.T[0] ) )
    ymin = int( np.min( stateSpace.T[1] ) )
    xmax = int( np.max( stateSpace.T[0] ) )
    ymax = int( np.max( stateSpace.T[1] ) )
    yTiledLen = np.ceil( float( ymax - ymin ) / float( yTiling ) )

    for i in range( xmin, xmax + 1 ):
        for j in range( ymin, ymax + 1 ):
            s = np.array( [i, j], dtype=np.float )
            if len( np.where( np.all( stateSpace == s, axis=1 ) )[0] ) == 0:
                continue
            phiInd = np.floor( float( i - xmin ) / float( xTiling ) ) * yTiledLen + np.floor( float( j - ymin ) / float( yTiling ) )
            indexMap[str( s )] = int( phiInd )

    dim = int( np.max( np.array( [indexMap[s] for s in indexMap] ) ) + 1 )

    def phi( s ):
        s = np.array( s, copy=True, dtype=np.float )
        if len( np.where( np.all( stateSpace == s, axis=1 ) )[0] ) == 0:
            raise Exception( 'State ' + str( s ) + ' not in MDP state space.' )
        p = np.zeros( dim )
        p[indexMap[str( s )]] = 1.0
        return p

    return phi, dim


def tiling2DActionStateBasisFunction( tabularMDP=None, stateSpace=None, actionSpace=None, tiling=1 ):
    '''
    @deprecated: Use tiledStateActionBinaryBasisFunction.
    Compute a tiled basis function. an be called in the following ways:
        
        tabularStateBasisFunction(tabularMDP)
        tabularStateBasisFunction(stateSpace=S, actionSpace=A)
        tabularStateBasisFunction(tabularMDP, tiling=1)
        tabularStateBasisFunction(stateSpace=S, actionSpace=A, tiling=1)
    
    @param tabularMDP: Tabular MDP.
    @param stateSpace: State space.
    @param actionSpace: Action space.
    @param tiling: Tile size, default 1
    
    @return: phi, dim. Phi is a function handle mapping a state to a numpy 
        array (vector). dim is the dimension of the returned array.
    '''
    if stateSpace == None:
        stateSpace = tabularMDP.getStateSpace()
    if actionSpace == None:
        actionSpace = tabularMDP.getActionSpace()

    phi, dim = tiling2DStateBasisFunction( stateSpace=stateSpace, tiling=tiling )
    dimAct = len( actionSpace ) * dim
    def phiAction( s, a ):
        aInd = int( np.where( actionSpace == a )[0][0] )
        p = np.zeros( dimAct )
        p[aInd * dim:( aInd + 1 ) * dim] = phi( s )
        return p

    return phiAction, dimAct



def get1DTiledBinaryBasisFunction( tileSize, domain, offset=0 ):
    '''
    Returns a 1D binary basis function for the given domain. The domain is 
    tiled into tiles of given size and the offset specifies the distance from 
    the minimum value in the domain to the first tile border. This also means 
    that offset < tileSize.
    
    @param tileSize: Length of a tile.
    @param domain: The domain of the function basis function.
    @param offset: Offset of the first tile border to the minimum element in 
        the domain. Default is 0.
        
    @return: 1D tiled binary basis function handle.    
    '''
    if offset >= tileSize:
        raise Exception( 'Offset ' + str( offset ) + ' has to be smaller than tile size ' \
                        + str( tileSize ) + ' and has to be non-negative.' )

    vmin = np.min( domain )
    vmax = np.max( domain )

    if offset > 0:
        tileOrigin = vmin + offset - tileSize
    else:
        tileOrigin = vmin
    numberOfTiles = int( np.floor( ( vmax - tileOrigin ) / tileSize + 1.0 ) )

    def phi( x ):
        ind = int( np.floor( ( x - tileOrigin ) / tileSize ) )
        b = np.zeros( numberOfTiles )
        b[ind] = 1.0
        return b

    return phi


def get1DLabelSetBinaryBasisFunction( labelSet, offset=None ):
    '''
    Calculate a binary basis function for a label set. The basis function returns
    a binary vector of the same length as the label set with a 1 set to the corresponding
    input element, all other fields are set to 0.
    
    @param labelSet: Label set, must be a numpy array.
    @param offset: Offset vector that can be added to the basis function vector.
    
    @return: Basis function handle.
    '''
    if offset == None:
        offset = np.zeros( len( labelSet ) )
    else:
        offset = np.array( offset )

    def phi( x ):
        sInd = np.where( x == labelSet )[0][0]
        p = np.zeros( len( labelSet ) )
        p[sInd] = 1.0
        return p

    return phi



def get1DPolynomialBasisFunction( degree, offset=None ):
    '''
    Calculate a 1D polynomial basis.
    
    @param degree: The degree of the polynomial.
    @param offset: Offset vector that is added to calculated polynomial basis function.
    
    @return: Basis function handle.
    '''
    exponents = np.array( range( 0, degree + 1 ) )
    if offset == None:
        offset = np.zeros( len( exponents ) )
    else:
        offset = np.array( offset )

    def phi( x ):
        return np.power( x, exponents ) + offset

    return phi

def calculateNDimBasisFunction( singleDimBasisFunction, ndimValue ):
    '''
    
    '''
    if not hasattr( singleDimBasisFunction, '__iter__' ):
        basisFnAr = [ singleDimBasisFunction for _ in range( len( ndimValue ) ) ]
        singleDimBasisFunction = basisFnAr

    phi = singleDimBasisFunction[0]( ndimValue[0] )
    for i in range( 1, len( singleDimBasisFunction ) ):
        phiNext = singleDimBasisFunction[i]( ndimValue[i] )
        phi = np.outer( phi, phiNext ).flatten()

    return phi

def tiledStateBinaryBasisFunction( stateSpace, tileSize, offset ):
    '''
    Tiled state binary basis function.
    
    @param stateSpace: MDP state space, each state must be a real valued array of the same length.
    @param tileSize: Tile size. If one number is given, the tile has the same length in all 
        dimensions. If an array is given, then each entry in the array specifies the tile length 
        for each dimension.
    @param offset: Tile offset. If one number is given, the tile has the same offset in all 
        dimensions. If an array is given, then each entry in the array specifies the tile offset 
        for each dimension.
        
    @return: Basis function handle.
    '''

    sdim = len( stateSpace[0] )
    if not hasattr( tileSize, '__iter__' ):
        tileSizeAr = [ tileSize for _ in range( sdim ) ]
        tileSize = tileSizeAr
    if not hasattr( offset, '__iter__' ):
        offsetAr = [offset for _ in range( sdim )]
        offset = offsetAr

    basisFunctionAr = []
    for i in range( sdim ):
        basisFunctionAr.append( get1DTiledBinaryBasisFunction( tileSize[i], stateSpace.T[i], offset[i] ) )

    def phi( s ):
        return calculateNDimBasisFunction( basisFunctionAr, s )
    return phi

def tiledStateActionBinaryBasisFunction( stateSpace, actionSpace, tileSize, offset ):
    '''
    Tiled state-action binary basis function. The action space is tiled with tiles of size 1.
    
    @param stateSpace: MDP state space, each state must be a real valued array of the same length.
    @param actionSpace: MDP action space, must be a numpy array, but can by any label set.
    @param tileSize: Tile size. If one number is given, the tile has the same length in all 
        dimensions. If an array is given, then each entry in the array specifies the tile length 
        for each dimension.
    @param offset: Tile offset. If one number is given, the tile has the same offset in all 
        dimensions. If an array is given, then each entry in the array specifies the tile offset 
        for each dimension.
    
    @return: Basis function handle.
    '''

    sdim = len( stateSpace[0] )
    if not hasattr( tileSize, '__iter__' ):
        tileSizeAr = [ tileSize for _ in range( sdim ) ]
        tileSize = tileSizeAr
    if not hasattr( offset, '__iter__' ):
        offsetAr = [offset for _ in range( sdim )]
        offset = offsetAr

    actionBasisFunction = get1DLabelSetBinaryBasisFunction( actionSpace )
    basisFunctionAr = []
    for i in range( sdim ):
        basisFunctionAr.append( get1DTiledBinaryBasisFunction( tileSize[i], stateSpace.T[i], offset[i] ) )

    def phi( s, a ):
        v = [ a, s ]

        phia = actionBasisFunction( a )
        phis = calculateNDimBasisFunction( basisFunctionAr, s )
        phiv = np.outer( phia, phis ).flatten()
        return phiv
    return phi

def tileCodedStateBinaryBasisFunction( stateSpace, tileSize, tilings ):
    '''
    This function computes multiple tiledStateBinaryBasisFunctions and overlays them with
    a random offset.
    
    @param stateSpace: State space
    @param tileSize: Size of the tiles (can be an array, see tiledStateBinaryBasisFunction)
    @param tilings: The number of tilings that should be overlayed.
    
    @return: Basis function handle.
    '''
    tilingfn = []
    sdim = len( stateSpace[0] )
    if not hasattr( tileSize, '__iter__' ):
        tileSizeAr = [ tileSize for _ in range( sdim ) ]
        tileSize = tileSizeAr
    np.array( tileSize )
    for _ in range( tilings ):
        offset = np.random.uniform( 0, 1, sdim )
        offset *= tileSize
        tilingfn.append( tiledStateBinaryBasisFunction( stateSpace, tileSize, offset ) )
    def phi( s ):
        p = tilingfn[0]( s )
        for f in tilingfn[1:]:
            p = np.append( p, f( s ) )
        return p
    return phi

def tileCodedStateActionBinaryBasisFunction( stateSpace, actionSpace, tileSize, tilings ):
    '''
    Same as tileCodedStateBinaryBasisFunction, but it crosses the basis function with an action
    space tile coder.
    
    @param stateSpace: State space
    @param actionSpace: Action space
    @param tileSize: Size of the tiles (can be an array, see tiledStateBinaryBasisFunction)
    @param tilings: The number of tilings that should be overlayed.
    
    @return: Basis function handle.
    '''
    actionBasisFunction = get1DLabelSetBinaryBasisFunction( actionSpace )
    stateBasisFunction = tileCodedStateBinaryBasisFunction( stateSpace, tileSize, tilings )
    def phi( s, a ):
        pa = actionBasisFunction( a )
        ps = stateBasisFunction( s )
        p = np.outer( pa, ps ).flatten()
        return p
    return phi


def polynomialStateBinaryBasisFunction( stateSpace, degree, offset ):
    '''
    Polynomial state basis function.
    
    @param stateSpace: MDP state space, each state must be a real valued array of the same length.
    @param degree: Degree of the polynomial. If one number is given, the degree of the polynomial 
        is the same for all dimensions. If an array is given, then each entry in the array specifies 
        the degree of the polynomial for each dimension.
    @param offset: Polynomial offset. If one number is given, the offset is added to each element 
        in the basis function vector. If an array is given, then each entry in the array specifies 
        the offset for the polynomial in each dimension.
        
    @return: Basis function handle.
    '''

    sdim = len( stateSpace[0] )
    if not hasattr( degree, '__iter__' ):
        degreeAr = [ degree for _ in range( sdim ) ]
        degree = degreeAr
    if not hasattr( offset, '__iter__' ):
        offsetAr = [offset for _ in range( sdim )]
        offset = offsetAr

    basisFunctionAr = []
    for i in range( sdim ):
        basisFunctionAr.append( get1DPolynomialBasisFunction( degree[i], offset ) )

    def phi( s ):
        return calculateNDimBasisFunction( basisFunctionAr, s )
    return phi

def polynomialStateActionBinaryBasisFunction( stateSpace, actionSpace, degree, offset ):
    '''
    Polynomial state-action basis function. The action space is tiled with tiles of size 1.
    
    @param stateSpace: MDP state space, each state must be a real valued array of the same length.
    @param actionSpace: MDP action space, must be a numpy array, but can by any label set.
    @param degree: Degree of the polynomial. If one number is given, the degree of the polynomial 
        is the same for all dimensions. If an array is given, then each entry in the array specifies 
        the degree of the polynomial for each dimension.
    @param offset: Polynomial offset. If one number is given, the offset is added to each element 
        in the basis function vector. If an array is given, then each entry in the array specifies 
        the offset for the polynomial in each dimension.
        
    @return: Basis function handle.
    '''

    sdim = len( stateSpace[0] )
    if not hasattr( degree, '__iter__' ):
        degreeAr = [ degree for _ in range( sdim ) ]
        degree = degreeAr
    if not hasattr( offset, '__iter__' ):
        offsetAr = [offset for _ in range( sdim )]
        offset = offsetAr

    actionBasisFunction = get1DLabelSetBinaryBasisFunction( actionSpace )
    basisFunctionAr = [ ]
    for i in range( sdim ):
        basisFunctionAr.append( get1DPolynomialBasisFunction( degree[i], offset ) )

    def phi( s, a ):
        phia = actionBasisFunction( a )
        phis = calculateNDimBasisFunction( basisFunctionAr, s )
        phiv = np.outer( phia, phis ).flatten()
        return phiv
    return phi

def basisFunctionMatrixState( basisFunction, mdp ):
    '''
    Computes a basis function matrix where each row corresponds to the basis function evaluated for a
    mdp state.
    
    @param basisFunction: State basis function.
    @param mdp: TabularMDP
    
    @return: Basis function matrix.
    '''
    phiDim = len( basisFunction( mdp.getStateSpace()[0] ) )
    phiMat = np.zeros( ( len( mdp.getStateSpace() ), phiDim ) )
    for i in range( len( mdp.getStateSpace() ) ):
        phiMat[i] = basisFunction( mdp.getStateSpace()[i] )
    return phiMat

def basisFunctionMatrixStateAction( basisFunction, mdp ):
    '''
    Computes a basis function matrix where each row corresponds to the basis function evaluated for a
    mdp state-action pair.
    
    @param basisFunction: State-action basis function.
    @param mdp: TabularMDP
    
    @return: Basis function matrix.
    '''
    phiDim = len( basisFunction( mdp.getStateSpace()[0], mdp.getActionSpace()[0] ) )
    saIt = mdp.getStateActionPairIterable()
    phiMat = np.zeros( ( len( saIt ), phiDim ) )
    for i in range( len( saIt ) ):
        sa = saIt[i]
        phiMat[i] = basisFunction( sa[0], sa[1] )
    return phiMat

def getTiledStateActionBasisFunction( mdp, tileNumPerStateDimension ):
    tileNumPerStateDimension = np.array( tileNumPerStateDimension )
    minS = np.array( map( lambda s: np.min( s ), mdp.getStateSpace().T ) )
    maxS = np.array( map( lambda s: np.max( s ), mdp.getStateSpace().T ) )

#     tileNumPerStateDimension = np.array([2.,2.,2.,2.])
    tileLen = ( maxS - minS ) / tileNumPerStateDimension
#     phiLen = int( np.prod(tileNumPerStateDimension) )
#     print 'number of features: ' + str(phiLen)

    def phis( s ):
        stripeInd = np.clip( np.floor( ( s - minS ) / tileLen - ( 10 ** -10 ) ), np.zeros( len( minS ) ), tileNumPerStateDimension - 1 )
        stripeInd = np.array( stripeInd, dtype=np.int )
        phiv = []

        for i in range( len( tileNumPerStateDimension ) ):
            stripe = np.zeros( tileNumPerStateDimension[i] )
            stripe[stripeInd[i]] = 1.0
    #         print str(i) + ':' + str(stripe)
            if len( phiv ) == 0:
                phiv = stripe
            else:
                phiv = np.outer( phiv, stripe ).flatten()
        return phiv

    actionSet = mdp.getActionSpace()
    def phia( a ):
        aInd = np.where( a == actionSet )[0][0]
        phiv = np.zeros( len( actionSet ) )
        phiv[aInd] = 1.0
        return phiv

    def phi( s, a ):
        ps = phis( s )
        pa = phia( a )
        return np.outer( pa, ps ).flatten()

    return phi

