import gp_class, gp_routines, kernels
import time
import numpy as np
import matplotlib.pyplot as plt

# A simple script to illustrate the basic
# features of the gps package. 

# Toy dataset:
n = 25
x = np.r_[ -5:5:1j*n ]
y = np.sin( x ) + 0.2*x
whitenoise = 0.2
e = whitenoise*np.random.randn( n )
data = y + e

# Create GP object:
gp_obj = gp_class.gp( which_type='full' ) # full rather than sparse GP
gp_obj.mfunc = None # zero mean function; otherwise point to user-defined mean function
gp_obj.mpars = {} # empty dict for mean function parameters
gp_obj.cfunc = kernels.sqexp # squared exponential covariance kernel
gp_obj.cpars = { 'amp':1, 'scale':2 } # covariance parameters
# Note: Users are encouraged to write their own covariance
# kernels to suit their specific task; the format for doing
# this is very straightforward - see the kernels.py module
# for examples and notes.

# Training inputs must be NxM array where M is
# the dimensionality of input space; here M=1:
gp_obj.xtrain = np.reshape( x, [ n, 1 ] )

# Same for training data:
gp_obj.dtrain = np.reshape( data, [ n, 1 ] )

# White noise error term: 
gp_obj.etrain = whitenoise
# Note that this can alternatively be set as an
# Nx1 array if different error terms are associated
# with different data points.

# Generate some random draws from the GP prior,
# assuming we haven't seen the data yet:
xmesh = np.reshape( np.r_[ -5:5:1j*400 ], [ 400, 1 ] )
emesh = None # i.e. set white noise to zero on draws
draws_unconditioned = gp_obj.random_draw( xmesh=xmesh, emesh=emesh, conditioned=False, \
                                          ndraws=4, plot_draws=True )

# Now do the same thing, but the the random draws
# taken from the GP conditioned on the data:
draws_conditioned = gp_obj.random_draw( xmesh=xmesh, emesh=emesh, conditioned=True, \
                                        ndraws=4, plot_draws=True )


# In practice, we would probably like to optimise
# the covariance parameters using the training data.
# We can do this by wrapping the GP log likelihood
# inside an optimiser, MCMC etc. Here's how to
# evaluate the log likelihood:
t1 = time.time()
logp = gp_obj.logp_builtin()
t2 = time.time()
print '\nlogp_builtin = {0}'.format( logp )
print 'time taken = {0:.5f} sec'.format( t2-t1 )

# Sometimes we might want to fix the covariance
# parameters, and optimise for the mean function
# parameters. In this case, the expensive matrix
# inversion needed to evaluate the GP likelihood
# only needs to be performed once, so subsequent
# evaluations of the likelihood are very quick.
# Here's how to do this:
cov_kwpars = gp_obj.prep_fixedcov() # does precomputations
t1 = time.time()
logp = gp_obj.logp_fixedcov( resids=gp_obj.dtrain, kwpars=cov_kwpars )
t2 = time.time()
print '\nlogp_fixedcov = {0}'.format( logp )
print 'time taken = {0:.5f} sec'.format( t2-t1 )
# Note: In a real problem, the 'resids' input is
# usually the difference between our training data
# and our model for the mean function. Here, seeing
# as we're using a zero mean function, we can assume
# this step has already been performed, i.e. the
# training data are already our model residuals.

# It's possible to evaluate the mean and covariance
# matrix of the GP:
mu, cov = gp_obj.meancov( xnew=xmesh, enew=gp_obj.etrain, conditioned=True )

# Or if you only want to evaluate the diagonal terms
# of the covariance matrix to save time, use:
mu, sig = gp_obj.predictive( xnew=xmesh, enew=gp_obj.etrain, conditioned=True )
# The 'sig' output is the square root of the covariance
# matrix diagonal, so it can be thought of as the 1-sigma
# predictive uncertainty.

plt.figure()
plt.errorbar( x, data, yerr=whitenoise, fmt='ok' )
plt.plot( xmesh.flatten(), mu.flatten(), '-b', lw=2 )
plt.fill_between( xmesh.flatten(), \
                  mu.flatten()-2*sig.flatten(), \
                  mu.flatten()+2*sig.flatten(), \
                  color=[ 0.9, 0.9, 0.9 ] )
plt.fill_between( xmesh.flatten(), \
                  mu.flatten()-1*sig.flatten(), \
                  mu.flatten()+1*sig.flatten(), \
                  color=[ 0.7, 0.7, 0.7 ] )
plt.title( 'Predictive distribution with 1-sigma and 2-sigma uncertainties shaded' )                  

