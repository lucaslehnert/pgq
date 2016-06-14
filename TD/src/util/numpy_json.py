'''
Created on Apr 23, 2015

This module contains functions to convert dictionaries containing numpy arrays
to JSON strings.

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''

import numpy as np
import json
import os

def numpyDictionaryToJson( dictionary, fileName=[] ):
    '''
    This function encodes the given dictionary as a JSON string. It uses a modified
    encoder to properly encode numpy arrays. To decode this formatting use jsonToNumpyDictionary.
    
    @param dictionary: A dictionary of basic data types and numpy arrays.
    @param fileName: Name of the file into which to write the JSON string.
    
    @return: Encoded JSON string or nothing if fileName is specified.
    
    @see: jsonToNumpyDictionary   
    '''
    class NumpyAwareJSONEncoder( json.JSONEncoder ):
        def default( self, obj ):
            if isinstance( obj, np.ndarray ) and obj.ndim == 1:
                return obj.tolist()
            elif isinstance( obj, np.ndarray ) and obj.ndim > 1:
                return { 'numpy.shape' : np.shape( obj ), 'numpy.ndarray' : np.reshape( obj, ( -1 ) ).tolist() }
            elif isinstance( obj, dict ) and len( obj.keys() ) > 1 and isinstance( obj.keys()[0], np.float64 ):
                return {'dict.keytype' : 'np.float64', 'dict' : dict}
            return json.JSONEncoder.default( self, obj )

    if fileName == []:
        return json.dumps( dictionary, cls=NumpyAwareJSONEncoder )
    else:
        with open( fileName, 'wb' ) as fp:
            json.dump( dictionary, fp, cls=NumpyAwareJSONEncoder )

def jsonToNumpyDictionary( jsonStr=[], fileName=[] ):
    '''
    This function decodes the given json string into a dictionary. It also supports decoding
    of numpy arrays.
    
    @param jsonStr: JSON string.
    @param fileName: File name to load JSON string from.
    
    @return: Dictionary decoded from JSON string.
    
    @see: numpyDictionaryToJson
    
    '''
    def is_number( s ):
        try:
            float( s )
            return True
        except ValueError:
            return False
    def as_numpyarr( dct ):
        if 'numpy.shape' in dct:
            shape = np.array( dct['numpy.shape'] )
            ar = np.array( dct['numpy.ndarray'] )
            return np.reshape( ar, shape )
        else:
            dct_float = {}
            for key in dct.keys():
                if is_number( key ):
                    dct_float[float( key )] = dct[key]
                else:
                    dct_float[key] = dct[key]
            return dct_float

    if fileName != []:
        with open( fileName, 'rb' ) as fp:
            return json.load( fp, object_hook=as_numpyarr )
    else:
        return json.loads( jsonStr, object_hook=as_numpyarr )

def mergeJSONFiles( fileNameList ):
    '''
    Merge the given JSON file list into one JSON file.
    
    @param fileNameList: List of file paths.
    
    @return: Merged dictionary.
    '''
    mergedDict = []
    for fileName in fileNameList:
        resJSONPart = jsonToNumpyDictionary( fileName=fileName )
        if isinstance( resJSONPart, list ):
            mergedDict += resJSONPart
        else:
            mergedDict.append( resJSONPart )
    return mergedDict

def loadJSONResults( resFile ):
    '''
    Load result files from the given file path.
    
    @param resFile: Result file or directory path.
    
    @return: Experimen results.
    '''
    if os.path.isfile( resFile ):
        experimentResults = jsonToNumpyDictionary( fileName=resFile )
    elif os.path.exists( resFile ) and os.path.isdir( resFile ):
        fileList = []
        for f in os.listdir( resFile ):
            if os.path.splitext( f )[1] == '.json':
                fileList.append( resFile + '/' + f )
        experimentResults = mergeJSONFiles( fileList )
    elif os.path.exists( os.path.splitext( resFile )[0] ) and os.path.isdir( os.path.splitext( resFile )[0] ):
        fileList = []
        for f in os.listdir( os.path.splitext( resFile )[0] ):
            if os.path.splitext( f )[1] == '.json':
                fileList.append( os.path.splitext( resFile )[0] + '/' + f )
        experimentResults = mergeJSONFiles( fileList )
    else:
        raise Exception( 'Cannot load from result file ' + str( resFile ) )
    return experimentResults

