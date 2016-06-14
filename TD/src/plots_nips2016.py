'''
Created on May 23, 2016

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)

Script to generate all plots from the NIPS 2016 paper.

'''

import matplotlib
matplotlib.use( 'agg' )
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import matplotlib.pyplot as plt

import numpy as np
import glob
import os
from util.numpy_json import loadJSONResults

experimentDir = '../data/'
plotDir = '../plot/'

if not os.path.exists( plotDir ):
    os.makedirs( plotDir )

def loadResults( globPath ):
    dataFiles = glob.glob( globPath )
    res = map( lambda df : loadJSONResults( df ), dataFiles )
    return res

def episodeMSPBE( resultDict ):
    episodeLengthMat = resultDict['results']['mspbe']
    episodeLengthMean = np.mean( episodeLengthMat, axis=0 )
    episodeLengthStd = np.std( episodeLengthMat, axis=0 )
    return episodeLengthMean, episodeLengthStd
def episodeQnorm( resultDict ):
    episodeLengthMat = resultDict['results']['qnorm']
    episodeLengthMean = np.mean( episodeLengthMat, axis=0 )
    episodeLengthStd = np.std( episodeLengthMat, axis=0 )
    return episodeLengthMean, episodeLengthStd

def episodeLength( resultDict ):
    episodeLengthMat = resultDict['results']['episodeLength']
    episodeLengthMean = np.mean( episodeLengthMat, axis=0 )
    episodeLengthStd = np.std( episodeLengthMat, axis=0 )
    return episodeLengthMean, episodeLengthStd

def concatenateExperiments( experimentList ):
    exp = {}
    exp['configuration'] = experimentList[0]['configuration']
    exp['experiment'] = experimentList[0]['experiment']
    exp['results'] = {}
    exp['results']['episodeLength'] = np.array( map( lambda e: e['results']['episodeLength'][0], experimentList ) )
    exp['results']['successfulRepeats'] = np.sum( map( lambda e: e['results']['successfulRepeats'], experimentList ) )
    exp['results']['thetaNorm'] = np.array( map( lambda e: e['results']['thetaNorm'][0], experimentList ) )
    return exp

def makeBairdPolts():
    global experimentDir, plotDir
    resQ = loadResults( experimentDir + 'baird/baird-sweeps-Q.json' )[0][0]
    resGQ = loadResults( experimentDir + 'baird/baird-sweeps-GQ.json' )[0][0]
    resPGQ = loadResults( experimentDir + 'baird/baird-sweeps-PGQ.json' )[0][0]

    plt.figure( figsize=( 4, 2.8 ) )
    stdInterval = 50

    resQ['results']['mspbe'] = map( lambda r : r[:200], resQ['results']['mspbeDiv'] )
    m, v = episodeMSPBE( resQ )
    plt.plot( range( len( m ) ), m, 'k', label='Q', linewidth=2 )
    plt.errorbar( range( len( m ) )[stdInterval::stdInterval], m[stdInterval::stdInterval], \
                 yerr=v[stdInterval::stdInterval], ecolor='k', fmt=None, linewidth=1.5 )

    m, v = episodeMSPBE( resGQ )
    plt.plot( range( len( m ) ), m, 'g', label='GQ', linewidth=2 )
    plt.errorbar( range( len( m ) )[stdInterval::stdInterval], m[stdInterval::stdInterval], \
                 yerr=v[stdInterval::stdInterval], ecolor='g', fmt=None, linewidth=1.5 )

    m, v = episodeMSPBE( resPGQ )
    plt.plot( range( len( m ) ), m, 'b', label='PGQ', linewidth=2 )
    plt.errorbar( range( len( m ) )[stdInterval::stdInterval], m[stdInterval::stdInterval], \
                 yerr=v[stdInterval::stdInterval], ecolor='b', fmt=None, linewidth=1.5 )

    plt.legend()
    plt.ylim( [0, 3000] )
    plt.ylabel( 'MSPBE' )
    plt.xlabel( 'Update' )
    plt.gcf().tight_layout()
    # plt.show()
    plt.savefig( plotDir + '/bd_all_sweep_mspbe.pdf' )

    plt.figure( figsize=( 4, 2.8 ) )
    stdInterval = 50

    resQ['results']['qnorm'] = map( lambda r : r[:200], resQ['results']['qnormDiv'] )
    m, v = episodeQnorm( resQ )
    plt.plot( range( len( m ) ), m, 'k', label='Q', linewidth=2 )
    plt.errorbar( range( len( m ) )[stdInterval::stdInterval], m[stdInterval::stdInterval], \
                 yerr=v[stdInterval::stdInterval], ecolor='k', fmt=None, linewidth=1.5 )

    m, v = episodeQnorm( resGQ )
    plt.plot( range( len( m ) ), m, 'g', label='GQ', linewidth=2 )
    plt.errorbar( range( len( m ) )[stdInterval::stdInterval], m[stdInterval::stdInterval], \
                 yerr=v[stdInterval::stdInterval], ecolor='g', fmt=None, linewidth=1.5 )

    m, v = episodeQnorm( resPGQ )
    plt.plot( range( len( m ) ), m, 'b', label='PGQ', linewidth=2 )
    plt.errorbar( range( len( m ) )[stdInterval::stdInterval], m[stdInterval::stdInterval], \
                 yerr=v[stdInterval::stdInterval], ecolor='b', fmt=None, linewidth=1.5 )

    plt.legend()
    plt.ylim( [0, 30] )
    plt.ylabel( '$|| \pmb{Q} ||_\infty$' )
    plt.xlabel( 'Update' )
    plt.gcf().tight_layout()
    # plt.show()
    plt.savefig( plotDir + '/bd_all_sweep_qnorm.pdf' )

def makeMountainCarPlots():
    global experimentDir, plotDir
    res = loadResults( experimentDir + 'mountaincar/mc_all_[0-9][0-9][0-9][0-9].json' )

    plt.figure( figsize=( 5, 3.4 ) )

    resQ = filter( lambda r : r['configuration']['agent'] == 'Q', res )[0]
    resGQ = filter( lambda r : r['configuration']['agent'] == 'GQ', res )[0]
    resPGQ = filter( lambda r : r['configuration']['agent'] == 'PGQ', res )[0]

    m, v = episodeLength( resQ )
    plt.plot( range( len( m ) ), m, 'k', label='Q' )
    plt.errorbar( range( len( m ) )[5::5], m[5::5], yerr=v[5::5], ecolor='k', fmt=None )

    m, v = episodeLength( resGQ )
    plt.plot( range( len( m ) ), m, 'g', label='GQ' )
    plt.errorbar( range( len( m ) )[5::5], m[5::5], yerr=v[5::5], ecolor='g', fmt=None )

    m, v = episodeLength( resPGQ )
    plt.plot( range( len( m ) ), m, 'b', label='PGQ' )
    plt.errorbar( range( len( m ) )[5::5], m[5::5], yerr=v[5::5], ecolor='b', fmt=None, capthick=1.5 )

    plt.legend()
    plt.ylim( [0, 10500] )
    plt.ylabel( 'Episode Length' )
    plt.xlabel( 'Episode' )
    plt.gcf().tight_layout()
    # plt.show()
    plt.savefig( plotDir + '/mc_all_episode_length.pdf' )

    plt.figure( figsize=( 5, 3.4 ) )

    resGQ = filter( lambda r : r['configuration']['agent'] == 'GQ', res )[0]
    resPGQ = filter( lambda r : r['configuration']['agent'] == 'PGQ', res )[0]

    m, v = episodeLength( resGQ )
    plt.plot( range( len( m ) ), m, 'g', label='GQ' )
    plt.errorbar( range( len( m ) )[5::5], m[5::5], yerr=v[5::5], ecolor='g', fmt=None )

    m, v = episodeLength( resPGQ )
    plt.plot( range( len( m ) ), m, 'b', label='PGQ' )
    plt.errorbar( range( len( m ) )[5::5], m[5::5], yerr=v[5::5], ecolor='b', fmt=None, capthick=1.5 )

    plt.legend()
    plt.ylim( [0, 10500] )
    plt.ylabel( 'Episode Length' )
    plt.xlabel( 'Episode' )
    # plt.show()
    plt.gcf().tight_layout()
    # plt.show()
    plt.savefig( plotDir + '/mc_GQ_PGQ_episode_length.pdf' )

def makeAcrobotPlots():
    global experimentDir, plotDir
    res = loadResults( experimentDir + 'acrobot/ac_all_1000_[0-9][0-9][0-9][0-9].json' )

    resQ = concatenateExperiments( filter( lambda e: e['configuration']['agent'] == 'Q', res ) )
    resGQ = concatenateExperiments( filter( lambda e: e['configuration']['agent'] == 'GQ', res ) )
    resPGQ = concatenateExperiments( filter( lambda e: e['configuration']['agent'] == 'PGQ', res ) )

    plt.figure( figsize=( 5, 3.4 ) )

    stdInt = 100

    m, v = episodeLength( resQ )
    plt.plot( range( len( m ) ), m, 'k', label='Q', alpha=0.4 )
    plt.errorbar( range( len( m ) )[stdInt::stdInt], m[stdInt::stdInt], yerr=v[stdInt::stdInt],
                 ecolor='k', fmt=None, capthick=2.5 )

    m, v = episodeLength( resGQ )
    plt.plot( range( len( m ) ), m, 'g', label='GQ', alpha=0.6 )
    plt.errorbar( range( len( m ) )[stdInt::stdInt], m[stdInt::stdInt], yerr=v[stdInt::stdInt],
                 ecolor='g', fmt=None, capthick=2.5 )

    m, v = episodeLength( resPGQ )
    plt.plot( range( len( m ) ), m, 'b', label='PGQ', alpha=0.6 )
    plt.errorbar( range( len( m ) )[stdInt::stdInt], m[stdInt::stdInt], yerr=v[stdInt::stdInt],
                 ecolor='b', fmt=None, capthick=2.5 )

    plt.legend( ncol=3 )
    plt.ylim( [0, 1800] )
    # plt.gca().set_yscale('log')
    plt.ylabel( 'Episode Length' )
    plt.xlabel( 'Episode' )
    plt.gcf().tight_layout()
    plt.show()
    plt.savefig( plotDir + '/ac_all_1000_episode_length.pdf' )

    plt.figure( figsize=( 5, 3.4 ) )

    stdInt = 100

    m, v = episodeLength( resGQ )
    plt.plot( range( len( m ) ), m, 'g', label='GQ', alpha=0.6 )
    plt.errorbar( range( len( m ) )[stdInt::stdInt], m[stdInt::stdInt], yerr=v[stdInt::stdInt],
                 ecolor='g', fmt=None, capthick=2.5 )

    m, v = episodeLength( resPGQ )
    plt.plot( range( len( m ) ), m, 'b', label='PGQ', alpha=0.6 )
    plt.errorbar( range( len( m ) )[stdInt::stdInt], m[stdInt::stdInt], yerr=v[stdInt::stdInt],
                 ecolor='b', fmt=None, capthick=2.5 )

    plt.legend( ncol=3 )
    plt.ylim( [0, 1800] )
    # plt.gca().set_yscale('log')
    plt.ylabel( 'Episode Length' )
    plt.xlabel( 'Episode' )
    plt.gcf().tight_layout()
    # plt.show()
    plt.savefig( plotDir + '/ac_GQ_PGQ_1000_episode_length.pdf' )

def main():
    makeBairdPolts()
    makeMountainCarPlots()
    makeAcrobotPlots()
    return

if __name__ == '__main__':
    main()
