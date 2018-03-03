import pdb, os, sys
import matplotlib.pyplot as plt
import numpy as np
try:
    import numexpr
    numexpr_installed = True
except:
    numexpr_installed = False
import scipy.spatial
from scipy.special import gamma,kv

# THIS MODULE IS VERY MUCH A WORK IN PROGRESS: THE AIM IS TO EVENTUALLY
# HAVE IT CONTAINING THE BASIC BUILDING BLOCK KERNELS THAT GET USED
# AGAIN AND AGAIN. IN PRACTICE, THE USER WILL OFTEN DEFINE THEIR OWN
# CUSTOM KERNELS IN THEIR ANALYSIS SCRIPTS; BUT PERHAPS THEY CAN USE THE
# ROUTINES DEFINED IN THIS MODULE TO HELP CONSTRUCT THOSE CUSTOM KERNELS.

"""
  This module contains definitions for a number of standard covariance functions.
  A properly defined covariance function will have the following format:
    - the first two arguments x and y will be both NxD array-like objects where N 
      is the number of data points and D is the dimensionality of each data point
    - **kwargs that contain the hyperparameters of the covariance function
    
  An important point is that the covariance function definitions here do not include
  a white noise term added to the diagonal entries under any circumstances. This is
  because it is added later where needed.

  Function definitions following these basic guidelines can be added to this module
  whenever the need arises.

  Disclaimer: At the time of writing, most of the contents of this module had been
  taken from Neale's GPKernelFunctions.py module and Suzanne's MyGPKernels.py module
  with appropriate syntax changes. I've changed a fair bit in the transfer, so
  don't assume any of them work as expected until they've been retested.
"""

# UNDER CONSTRUCTION - most of the function definitions below need to be updated to the
# new syntax with more explicit parameter definitions rather than just 'theta' etc; there's
# probably quite a bit of other stuff that can be tidied up and tested as well as this.


########################################################################################

def sqexp( x, y, **cpars ):
    """
    Squared exponential kernel for 1D input.
    """

    amp = cpars['amp']
    scale = cpars['scale']

    x = np.matrix( x )/scale
    if y==None:
        n = np.shape( x )[0]
        cov = ( amp**2. ) + np.zeros( n )
        cov = np.reshape( cov, [ n, 1 ] )
    else:
        ny = np.shape( y )[0]
        y = np.matrix( y )/scale
        D2 = scipy.spatial.distance.cdist( x, y, 'sqeuclidean' )
        #cov = ( amp**2. ) * np.exp( -0.5 * D2 )
        cov = ( amp**2. ) * numexpr.evaluate( 'exp(-0.5 * D2)' )

    return cov


def matern32_invL( x, y, **cpars ):
  """
  Matern kernel with nu = 3/2 for 1D input.
  """
  
  amp = cpars['amp']
  iscale = cpars['iscale']

  x = np.matrix( x )
  if y==None:
      n = np.shape( x )[0]
      cov = ( amp**2. ) + np.zeros( n )
      cov = np.reshape( cov, [ n, 1 ] )
  else:
      y = np.matrix( y )
      D = scipy.spatial.distance.cdist( x, y, 'euclidean')
      arg = numexpr.evaluate( 'sqrt( 3 )*D*iscale' )
      poly_term = numexpr.evaluate( '1. + arg' )
      exp_term = numexpr.evaluate( 'exp( -arg )' )
      cov = numexpr.evaluate( '( amp**2 )*poly_term*exp_term' )

  return cov



def sqexp_invL( x, y, **cpars ):
    """
    Squared exponential kernel for 1D input.
    """

    amp = cpars['amp']
    iscale = cpars['iscale']

    x = np.matrix( x )*np.sqrt( iscale )
    if y==None:
        n = np.shape( x )[0]
        cov = ( amp**2. ) + np.zeros( n )
        cov = np.reshape( cov, [ n, 1 ] )
    else:
        ny = np.shape( y )[0]
        y = np.matrix( y )*np.sqrt( iscale )
        D2 = scipy.spatial.distance.cdist( x, y, 'sqeuclidean' )
        #cov = ( amp**2. ) * np.exp( -0.5 * D2 )
        cov = ( amp**2. ) * numexpr.evaluate( 'exp(-0.5 * D2)' )

    return cov


def sqexp_ard( x, y, **cpars ):
    """
    Squared exponential kernel with ARD for N-dimension inputs.
    """
    amp = cpars['amp']
    scales = np.array( cpars['scale'] )
    x = np.matrix( x )
    if y==None:
        n = np.shape( x )[0]
        cov = ( amp**2. ) + np.zeros( n )
        cov = np.reshape( cov, [ n, 1 ] )
    else:
        y = np.matrix( y )
        v = np.matrix( np.diag( 1./scales ) )
        x = x * v
        y = y * v
        D2 = scipy.spatial.distance.cdist( x, y, 'sqeuclidean' )
        cov = ( amp**2. ) * np.exp( -0.5 * D2 )
    return cov


def sqexp_invL_ard_numpy( x, y, **cpars ):
    """
    Squared exponential kernel with ARD for N-dimension inputs.

    k(x1,x2) = A*exp[ -0.5*sum(Di/Li)^2 ]
    
    where x1 and x2 are two NxM arrays with each column representing
    a different input vector, Di is the Euclidean norm between the ith
    input vectors of x1 and x2, and Li is the corresponding covariance
    length scale. In terms of cpars:

    'amp' - Covariance amplitude A
    'scale' - Array containing the inverse covariance length scales of
              the form 0.5/(Li^2).
    """

    amp = cpars['amp']
    scales = np.array( cpars['iscale'] )

    x = np.matrix( x )
    if y==None:
        n = np.shape( x )[0]
        cov = ( amp**2. ) + np.zeros( n )
        cov = np.reshape( cov, [ n, 1 ] )
    else:
        y = np.matrix( y )
        v = np.matrix( np.sqrt( np.diag( scales ) ) )
        x = x * v
        y = y * v
        D2 = scipy.spatial.distance.cdist( x, y, 'sqeuclidean' )
        cov = ( amp**2. ) * np.exp( -D2 )

    return cov

def sqexp_invL_ard( x, y, **cpars ):
    """
    Squared exponential kernel with ARD for N-dimension inputs.

    k(x1,x2) = A*exp[ -0.5*sum(Di/Li)^2 ]
    
    where x1 and x2 are two NxM arrays with each column representing
    a different input vector, Di is the Euclidean norm between the ith
    input vectors of x1 and x2, and Li is the corresponding covariance
    length scale. In terms of cpars:

    'amp' - Covariance amplitude A
    'scale' - Array containing the inverse covariance length scales of
              the form 0.5/(Li^2).
    """

    if numexpr_installed==False:

        cov = sqexp_invL_ard_numpy( x, y, **cpars )

    else:

        amp = cpars['amp']
        amp2 = numexpr.evaluate( 'amp**2.' )
        #amp2 = cpars['amp2']
        iscales = np.array( cpars['iscale'] )

        x = np.matrix( x )
        if y==None:
            n = np.shape( x )[0]
            cov = amp2 + np.zeros( n )
            cov = np.reshape( cov, [ n, 1 ] )
        else:
            y = np.matrix( y )
            sqrt_iscales = numexpr.evaluate( 'sqrt( iscales )' )
            v = np.matrix( np.diag( sqrt_iscales ) )
            x = x*v # cannot do matrix multiplication with numexpr
            y = y*v # cannot do matrix multiplication with numexpr
            # GENERAL COMMENT: I imagine numexpr should also be able to 
            # speed up the scipy.spatial.distance.cdist() calls... but 
            # will need to think about how to implement this...
            D2 = scipy.spatial.distance.cdist( x, y, 'sqeuclidean' )
            cov = numexpr.evaluate( 'amp2*exp( -D2 )' )

    return cov


def matern32_invL_ard( x, y, **cpars ):
    """
    Matern v=3/2 kernel with ARD for N-dimension inputs.
    'amp' - Covariance amplitude A
    'scale' - Array containing the inverse covariance length scales.
    """

    if numexpr_installed==False:

        cov = matern32_invL_ard_numpy( x, y, **cpars )

    else:

        amp = cpars['amp']
        amp2 = numexpr.evaluate( 'amp**2.' )
        #amp2 = cpars['amp2']
        iscales = np.array( cpars['iscale'] )

        x = np.matrix( x )
        if y==None:
            n = np.shape( x )[0]
            cov = amp2 + np.zeros( n )
            cov = np.reshape( cov, [ n, 1 ] )
        else:
            y = np.matrix( y )
            sqrt_iscales = numexpr.evaluate( 'sqrt( iscales )' )
            v = np.matrix( np.diag( sqrt_iscales ) )
            x = x*v # cannot do matrix multiplication with numexpr
            y = y*v # cannot do matrix multiplication with numexpr
            # GENERAL COMMENT: I imagine numexpr should also be able to 
            # speed up the scipy.spatial.distance.cdist() calls... but 
            # will need to think about how to implement this...
            D = scipy.spatial.distance.cdist( x, y, 'euclidean' )
            arg = numexpr.evaluate( 'sqrt( 3. )*D' )
            #poly_term = numexpr.evaluate( '1. + arg + ( ( arg**2. )/3. )' )
            poly_term = numexpr.evaluate( '1. + arg' )
            exp_term = numexpr.evaluate( 'exp( -arg )' )
            cov = numexpr.evaluate( 'amp2*poly_term*exp_term' )

    return cov


def matern32_invL_ard_numpy( x, y, **cpars ):
    """
    Matern v=3/2 kernel with ARD for N-dimension inputs.
    'amp' - Covariance amplitude A
    'scale' - Array containing the inverse covariance length scales.
    """

    amp2 = cpars['amp']**2.
    iscales = np.array( cpars['iscale'] )

    x = np.matrix( x )
    if y==None:
        n = np.shape( x )[0]
        cov = amp2 + np.zeros( n )
        cov = np.reshape( cov, [ n, 1 ] )
    else:
        y = np.matrix( y )
        v = np.matrix( np.diag( np.sqrt( iscales ) ) )
        x = x*v
        y = y*v
        D = scipy.spatial.distance.cdist( x, y, 'euclidean' )
        poly_term = 1. + np.sqrt( 3. )*D
        exp_term = np.exp( -arg )
        cov = amp2*poly_term*exp_term

    return cov



########################################################################################
# TODO: matern12_ARD, matern32_ARD, matern52_ARD

def matern12( x, y, **cpars ):
  """
  Matern kernel with nu = 1/2 (i.e. Ornstein-Uhlenbeck process)
  for 1D input.
  """
  
  amp = cpars['amp']
  scale = cpars['scale']

  x = np.matrix( x )
  if y==None:
      n = np.shape( x )[0]
      cov = ( amp**2. ) + np.zeros( n )
      cov = np.reshape( cov, [ n, 1 ] )
  else:
      y = np.matrix( y )
      D = scipy.spatial.distance.cdist( x, y, 'euclidean')
      cov = ( amp**2. )*np.exp( -D/scale )

  return cov

def matern32( x, y, **cpars ):
  """
  Matern kernel with nu = 3/2 for 1D input.
  """
  
  amp = cpars['amp']
  scale = cpars['scale']

  x = np.matrix( x )
  if y==None:
      n = np.shape( x )[0]
      cov = ( amp**2. ) + np.zeros( n )
      cov = np.reshape( cov, [ n, 1 ] )
  else:
      y = np.matrix( y )
      D = scipy.spatial.distance.cdist( x, y, 'euclidean')
      arg = np.sqrt( 3 ) * D / scale
      poly_term = ( 1. + arg ) 
      exp_term = np.exp( -arg )
      cov = ( amp**2 ) * poly_term * exp_term

  return cov

def matern32_ard( x, y, **cpars ):
    """
    Multi-dimensional inputs with individual length scales, Matern32.
    """

    amp = cpars['amp']
    scales = np.array( cpars['scale'] )
    
    x = np.matrix( x )
    if y==None:
        n = np.shape( x )[0]
        cov = ( amp**2. ) + np.zeros( n )
        cov = np.reshape( cov, [ n, 1 ] )
    else:
        y = np.matrix( y )
        v = np.matrix( np.diag( 1./scales ) )
        x = x * v
        y = y * v
        D = scipy.spatial.distance.cdist( x, y, 'euclidean' )
        arg = np.sqrt( 3 )*D
        #poly_term = 1. + arg + ( ( arg**2. )/3. )
        poly_term = 1. + arg
        exp_term = np.exp( -arg )
        cov = ( amp**2. )*poly_term*exp_term

    return cov
 
def matern52( x, y, **cpars ):
  """
  Matern kernel with nu = 5/2
  """

  amp = cpars['amp']
  scale = cpars['scale']

  x = np.matrix( x )
  if y==None:
      n = np.shape( x )[0]
      cov = ( amp**2. ) + np.zeros( n )
      cov = np.reshape( cov, [ n, 1 ] )
  else:
      y = np.matrix( y )
      D = scipy.spatial.distance.cdist( x, y, 'euclidean')
      arg = np.sqrt( 5 ) * D / scale
      poly_term = ( 1. + arg + ( ( arg**2.)/3. ) )
      exp_term = np.exp( -arg )
      cov = ( amp**2 ) * poly_term * exp_term

  return cov


def matern52_ard( x, y, **cpars ):
    """
    Multi-dimensional inputs with individual length scales, Matern52.
    """

    amp = cpars['amp']
    scales = np.array( cpars['scale'] )
    
    x = np.matrix( x )
    if y==None:
        n = np.shape( x )[0]
        cov = ( amp**2. ) + np.zeros( n )
        cov = np.reshape( cov, [ n, 1 ] )
    else:
        y = np.matrix( y )
        v = np.matrix( np.diag( 1./scales ) )
        x = x * v
        y = y * v
        D = scipy.spatial.distance.cdist( x, y, 'euclidean' )
        arg = np.sqrt( 5 )*D
        poly_term = 1. + arg + ( ( arg**2. )/3. )
        exp_term = np.exp( -arg )
        cov = ( amp**2. )*poly_term*exp_term

    return cov


########################################################################################
# Periodic covariance functions:


def SinPowExp( X, Y, amp=None, scale=None, power=None, period=None ):
  """
  Periodic covariance function: squared exponential of sinusoid.
  """
  D = ( EuclideanDist(X, Y) )**power
  K = ( amp**2 ) * np.exp( - 2 * np.sin( np.pi*D/period )**2 / scale**2 )
  return np.matrix(K)


def SinRatQuad( X, Y, amp=None, scale=None, index=None, period=None ):
  """
  Periodic covariance function: rational quadratic of sinusoid.
  """
  D = EuclideanDist(X, Y)
  D2 = np.sin( np.pi * D / period )**2
  K = ( amp**2 ) * (1 + D2 / (2 * index * scale**2) )**(-index)
  return np.matrix(K)


########################################################################################
# Quasi-periodic covariance functions:


def SinSqExp_SqExp( X, Y, amp=None, period=None, p_scale=None, ap_scale=None ):
  """
  Quasi periodic covariance function (SinSE times squared exponential decay).
  """
  D = GP.EuclideanDist(X,Y)
  D2 = GP.EuclideanDist2(X,Y)
  K_SinSqExp = np.exp( - 2 * ( np.sin( np.pi*D / period )**2 ) / ( p_scale**2 ) )
  K_SqExp = np.exp( - D2 / (2 * ( ap_scale**2 ) ) )
  K = ( amp**2 ) * K_SinSqExp * K_SqExp
  return np.matrix(K)


def SinSqExp_RatQuad( X, Y, theta ):
  """
  Quasi periodic covariance function (SinSE times rational quadratic decay).
  """
  
  amplitude, period, p_scale, ap_index, ap_scale = theta[:5]
  # Calculate distance matrices
  D = GP.EuclideanDist(X, Y)
  D2 = GP.EuclideanDist2(X, Y)
  # Calculate covariance matrix
  SinSE_term = np.exp( - 2 * np.sin(np.pi*D/period)**2 / p_scale**2 )
  RQ_term = (1 + D2 / (2 * ap_index * ap_scale**2))**(-ap_index)
  K = amplitude**2 * SinSE_term * RQ_term

  return np.matrix(K)


def SinSqExp_RatQuad_pSqExp( X, Y, theta ):
  """
  Quasi periodic covariance function (SinSE times rational quadratic decay + SE).
  """
  
  amplitude, period, p_scale, ap_index, ap_scale, add_amp, add_scale = theta[:7]
  # Calculate distance matrices
  D = GP.EuclideanDist(X, Y)
  D2 = GP.EuclideanDist2(X, Y)
  # Calculate covariance matrix
  SinSE_term = np.exp( - 2 * np.sin(np.pi*D/period)**2 / p_scale**2 )
  RQ_term = (1 + D2 / (2 * ap_index * ap_scale**2))**(-ap_index)
  K = amplitude**2 * SinSE_term * RQ_term
  # Add other SE term
  K += add_amp**2 * np.exp( - D2 / 2. / add_scale**2 )

  return np.matrix(K)


########################################################################################
# Auxilary functions to compute euclidean distances:


def EuclideanDist( X1, X2, v=None ):
  """
  Calculate the distance matrix for 2 data matricies
  X1 - n x D input matrix
  X2 - m x D input matrix
  v - weight vector
  D - output an n x m matrix of dist = sqrt( Sum_i (1/l_i^2) * ||x_i - x'_i|| )
  
  """

  if np.ndim( X1 )==1:
      X1 = np.reshape( X1, [ len( X1 ), 1 ] )
  if np.ndim( X2 )==1:
      X2 = np.reshape( X2, [ len( X2 ), 1 ] )

  X1,X2 = np.matrix(X1), np.matrix(X2)
  
  if v != None: #scale each coord in Xs by the weight vector
    V = np.abs(np.matrix( np.diag(1./v) ))
    X1 = X1 / V
    X2 = X2 / V
  
  #calculate euclidean distance (after weighting)
  D = scipy.spatial.distance.cdist( X1, X2, 'euclidean')

  return D

def EuclideanDist2( X1, X2, v=None ):
  """
  Calculate the distance matrix squared for 2 data matricies
  X1 - n x D input matrix
  X2 - m x D input matrix
  v - weight vector
  D2 - output an n x m matrix of dist^2 = Sum_i (1/l_i^2) * (x_i - x'_i)^2

  Adapted from NG's code...

  """

  if np.ndim( X1 )==1:
      X1 = np.reshape( X1, [ len( X1 ), 1 ] )
  if np.ndim( X2 )==1:
      X2 = np.reshape( X2, [ len( X2 ), 1 ] )
  
  #ensure inputs are in matrix form
  X1,X2 = np.matrix(X1), np.matrix(X2)
  
  if v != None: #scale each coord in Xs by the weight vector
    V = np.abs(np.matrix( np.diag(v) ))
    X1 = X1 * V
    X2 = X2 * V
  
  #calculate sqaured euclidean distance (after weighting)
  D2 = scipy.spatial.distance.cdist( X1, X2, 'sqeuclidean' )
  
  return D2

def EuclideanDist2_ORIG( X1, X2, v=None ):
  """
  Calculate the distance matrix squared for 2 data matricies
  X1 - n x D input matrix
  X2 - m x D input matrix
  v - weight vector
  D2 - output an n x m matrix of dist^2 = Sum_i (1/l_i^2) * (x_i - x'_i)^2
  
  TE's original....

  """

  if np.ndim( X1 )==1:
      X1 = np.reshape( X1, [ len( X1 ), 1 ] )
  if np.ndim( X2 )==1:
      X2 = np.reshape( X2, [ len( X2 ), 1 ] )
  
  #ensure inputs are in matrix form
  X1,X2 = np.matrix(X1), np.matrix(X2)
  
  if v != None: #scale each coord in Xs by the weight vector
    V = np.abs(np.matrix( np.diag(v) ))
    X1 = X1 * v
    X2 = X2 * v
  
  #calculate sqaured euclidean distance (after weighting)
  D2 = scipy.spatial.distance.cdist( X1, X2, 'sqeuclidean' )
  
  return D2


####################################################################################################

