import sys, os, pdb, time
import numpy as np
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt

PERTURB = 1e-7

def random_draw( gp_obj, xmesh=None, emesh=None, conditioned=True, perturb=PERTURB, ndraws=5, \
                 plot_draws=True, mesh_dim=0, lw=3 ):
    """
    SUMMARY

    Draws one or more random realisations from the gp and (optionally) plots them,
    along with the mean function (black dashed line) and 1- and 2-sigma uncertainty
    regions (shaded grey regions).
   

    CALLING
    
    draws = random_draw( gp_obj, xmesh=None, emesh=None, conditioned=True, perturb=PERTURB, \
                         ndraws=5, plot_draws=True, mesh_dim=0, lw=3 )

    INPUTS

    'xmesh' [KxD array] - input locations for the random draw points; if set to
        None (default), a fine grid spanning the xtrain range will be used.
    'emesh' [float] -  white noise value for the random draw points; if set to
        None (default) or zero, then this will be set to the value of the perturb
        variable for numerical stability.
    'conditioned' [bool] - if set to True (default), the GP will be trained on the 
        training data stored in the object; otherwise, it will be drawn from the
        unconditioned prior.
    'perturb' [float] - small perturbation to be added to the covariance diagonal for
        numerical stability if the white noise errors are set to None/zero.
    'ndraws' [integer] - the number of random draws to be made.
    'plot_draws' [bool] - if set to True, the random draws will be plotted.
    'mesh_dim' [integer] - for cases where D>1 (i.e. multidimensional input), a single
        input dimension must be specified for the mesh to span; the other input 
        variables will be held fixed to the corresponding median values in the training
        data set.
    'lw' [integer] - thickness of plot lines.

    OUTPUT

    'draws' [list] - a list containing the separate random draws from the GP.
    """
    
    xtrain = gp_obj.xtrain
    dtrain = gp_obj.dtrain
    etrain = gp_obj.etrain
    if xtrain!=None:
        n = np.shape( xtrain )[0]
        d = np.shape( xtrain )[1]    
    if xmesh==None:
        nmesh = 1000
        xmesh = np.r_[ xtrain.min() : xtrain.max() : 1j*nmesh ]
        xmesh = np.reshape( xmesh, [ nmesh, 1 ] )
    else:
        nmesh = np.shape( xmesh )[0]
    if conditioned==True:
        print '\nDrawing from GP posterior (i.e. after being trained on data set)'
        title_str = 'posterior (i.e. trained)'
    else:
        print '\nDrawing from GP prior (i.e. not trained on any data set)'
        title_str = 'prior (i.e. untrained)'

    mu, cov = meancov( gp_obj, xnew=xmesh, enew=emesh, conditioned=conditioned, perturb=perturb )
    sig = np.sqrt( np.diag( cov ).flatten() )
    mu = mu.flatten()
    sig = sig.flatten()
    xmesh_i = xmesh[:,mesh_dim].flatten()
    if plot_draws==True:
        fig = plt.figure()
        ax = fig.add_axes( [ 0.05, 0.05, 0.9, 0.9 ] )
        zorder0 = 0
        ax.fill_between( xmesh_i, mu-2*sig, mu+2*sig, color=[ 0.8, 0.8, 0.8 ], zorder=zorder0 )
        zorder0 = 1
        ax.fill_between( xmesh_i, mu-1*sig, mu+1*sig, color=[ 0.6, 0.6, 0.6 ], zorder=zorder0 )
        zorder0 = 2
        ax.plot( xmesh_i, mu, ls='--', c='g', lw=2, zorder=zorder0 )
        ax.set_title('%i random GP draws - %s' % ( ndraws, title_str ) )
        # Draw random samples from the GP:
        colormap = matplotlib.cm.cool
        colormap = plt.cm.ScalarMappable( cmap=colormap )
        colormap.set_clim( vmin=0, vmax=1 )
        line_colors = np.r_[ 0.05 : 0.95 : 1j*ndraws ]
        ax.set_xlim( [ xmesh_i.min(), xmesh_i.max() ] )

    draws = []
    for i in range( ndraws ):
        print ' drawing %i of %i on a mesh of %i points' % ( i+1, ndraws, nmesh )
        draw = np.random.multivariate_normal( mu, cov )
        draws += [ draw ]
        if plot_draws==True:
            color = colormap.to_rgba( line_colors[i] )
            zorder0 = 3
            ax.plot( xmesh_i, draw, ls='-', c=color, lw=lw, zorder=1 )
    if ( plot_draws==True )*( conditioned==True ):
        dtrain = dtrain.flatten()
        zorder0 = 4
        xtrain_i = xtrain[:,mesh_dim].flatten()
        if np.ndim( etrain )==0:
            if ( etrain==0 )+( etrain==None ):
                plot_errs = False
            else:
                plot_errs = True
                errs = etrain*np.ones( n )
        else:
            if ( np.ndim( etrain )==2 ):
                etrain = etrain.flatten()
            if ( np.all( etrain )==None )+( np.all( etrain )==0 ):
                plot_errs = False
            else:
                plot_errs = True
                errs = etrain
        if plot_errs==False:
            ax.plot( xtrain_i, dtrain, fmt='o', mec='k', mfc='k', zorder=zorder0 )
        else:
            ax.errorbar( xtrain_i, dtrain, yerr=errs, fmt='o', mec='k', mfc='k', ecolor='k', \
                         capsize=0, elinewidth=2, barsabove=True, zorder=zorder0 )
    
    return draws


def meancov( gp_obj, xnew=None, enew=None, conditioned=True, perturb=PERTURB ):
    """
    SUMMARY
    
    Returns the mean and full covariance of a gp at the locations of xnew, with
    random errors enew. If conditioned==True, the gp will be conditioned on the
    training data stored in the gp_obj. If etrain==None or etrain==0 (stored within
    gp_obj), a perturbation term of magnitude perturb will be added to the diagonal
    entries of the training covariance matrix before it is inverted for numerical
    stability.

    CALLING:
    
    mu, cov = meancov( gp_obj, xnew=None, enew=None, conditioned=True, perturb=PERTURB )

    INPUTS
    
    'gp_obj' [gp class object], containing:
              'mfunc', 'cfunc' [functions] - mean and covariance functions.
              'mpars', 'cpars' [dictionaries] - mean and covariance function parameters. 
              'xtrain' [NxD array] - training data input locations.
              'dtrain' [Nx1 array] - training data values.
              'etrain' [float] - white noise value for the training data points.
    'xnew' [PxD array] - input locations for the mean and covariance to be evaluated at;
          if set to None (default), the values for xtrain will be used.
    'enew' [float] - white noise value to be incorporated into the covariance diagonal;
          if set to None (default) or zero, it will be set to the value of the perturb
          variable for numerical stability.
    'conditioned' [bool] - if set to True (default), the gp will be trained on the
          training data stored in the object.
    'perturb' [float] - small perturbation to be added to the covariance diagonal for
          numerical stability if the white noise errors are set to None/zero.

    OUTPUT
    
    'mu' [Px1 array] - gp mean function values.
    'cov' [PxP array] - gp covariance values.
    """

    mfunc = gp_obj.mfunc
    mpars = gp_obj.mpars
    cfunc = gp_obj.cfunc
    cpars = gp_obj.cpars
    xtrain = gp_obj.xtrain
    dtrain = gp_obj.dtrain
    etrain = gp_obj.etrain
    if mfunc==None:
        mfunc = zero_mfunc
    if mpars==None:
        mpars = {}
    if cpars==None:
        cpars = {}
    if xnew==None:
        xnew = xtrain
        conditioned = False

    # The number of predictive points:
    p = np.shape( xnew )[0]

    if np.ndim( enew )==0:
        if ( enew==None )+( enew==0 ):
            enew = perturb*np.ones( p )
        else:
            enew = enew*np.ones( p )
    elif ( np.all( enew )==None )+( np.all( enew )==0 ):
        enew = perturn*np.ones( p )
    elif ( np.ndim( enew )==2 ):
        enew = enew.flatten()
    
    mnew = mfunc( xnew, **mpars ).flatten()
    knn = cfunc( xnew, None, **cpars ).flatten()

    # Evaluate the covariance matrix block for the new points:
    Kp = cfunc( xnew, xnew, **cpars ) + ( enew**2. ) * np.eye( p )

    # Evaluate the mean and covariance, which will require extra
    # work if the GP is to be conditioned on the training data:
    if conditioned==True:
        n = np.shape( xtrain )[0]
        if np.ndim( etrain )==0:
            if ( etrain==None )+( etrain==0 ):
                etrain = perturb*np.ones( n )
            else:
                etrain = etrain*np.ones( n )
        elif ( np.all( etrain )==None )+( np.all( etrain )==0 ):
            etrain = perturn*np.ones( n )
        elif ( np.ndim( etrain )==2 ):
            etrain = etrain.flatten()
        mtrain = mfunc( xtrain, **mpars )
        rtrain = np.matrix( dtrain.flatten() - mtrain.flatten() ).T
        mnew = np.matrix( mnew ).T
        Kn = np.matrix( cfunc( xtrain, xtrain, **cpars ) + np.diag( etrain**2. ) )
        Knp = np.matrix( cfunc( xtrain, xnew, **cpars ) )
        Kp = np.matrix( Kp )

        # Use Cholesky decompositions to efficiently calculate the mean:
        L = np.linalg.cholesky( Kn )
        Kninv_rtrain = np.matrix( scipy.linalg.lu_solve( scipy.linalg.lu_factor( Kn ), rtrain ) )
        mu = np.array( mnew + Knp.T * Kninv_rtrain ).flatten()

        # Now do similar for the covariance matrix:
        Linv_Knp = np.matrix( scipy.linalg.lu_solve( scipy.linalg.lu_factor( L ), Knp ) )
        KnpT_LLinv_Knp = Linv_Knp.T * Linv_Knp
        cov = np.array( Kp - KnpT_LLinv_Knp )

    else:
        mu = mnew
        cov = Kp

    mu = np.reshape( mu, [ p, 1 ] )
    return mu, cov


def predictive( gp_obj, xnew=None, enew=None, conditioned=True, perturb=PERTURB ):
    """
    SUMMARY
    
    Returns the predictive mean and standard deviation of a gp.  If conditioned==True,
    the gp will be conditioned on the training data stored in the gp_obj. If
    etrain==None or etrain==0 (stored within gp_obj), a perturbation term of magnitude
    perturb will be added to the diagonal entries of the training covariance matrix
    before it is inverted for numerical stability. This routine is very similar to
    meancov, except that it only calculates the diagonal entries of the conditioned
    gp's covariance matrix to save time.

    CALLING:
    
    mu, sig = predictive( gp_obj, xnew=None, enew=None, conditioned=True, perturb=PERTURB )

    INPUTS:
    
    'gp_obj' [gp class object], containing:
              'mfunc', 'cfunc' [functions] - mean and covariance functions.
              'mpars', 'cpars' [dictionaries] - mean and covariance function parameters. 
              'xtrain' [NxD array] - training data input locations.
              'dtrain' [Nx1 array] - training data values.
              'etrain' [float] - white noise value for the training data points.
    'xnew' [PxD array] - input locations for the mean and covariance to be evaluated at;
          if set to None (default), the values for xtrain will be used.
    'enew' [float] - white noise value to be incorporated into the covariance diagonal;
          if set to None (default) or zero, it will be set to the value of the perturb
          variable for numerical stability.
    'conditioned' [bool] - if set to True (default), the gp will be trained on the
          training data stored in the object.
    'perturb' [float] - small perturbation to be added to the covariance diagonal for
          numerical stability if the white noise errors are set to None/zero.

    OUTPUT:
    
    'mu' [Px1 array] - gp mean function values.
    'sig' [Px1 array] - 1-sigma marginalised uncertainties, i.e. the square roots of
          the entries along the diagonal of the full covariance matrix.
    """

    mfunc = gp_obj.mfunc
    mpars = gp_obj.mpars
    cfunc = gp_obj.cfunc
    cpars = gp_obj.cpars
    xtrain = gp_obj.xtrain
    dtrain = gp_obj.dtrain
    etrain = gp_obj.etrain
    n = np.shape( xtrain )[0]
    if mfunc==None:
        mfunc = zero_mfunc
    if mpars==None:
        mpars = {}
    if cpars==None:
        cpars = {}
    if xnew==None:
        xnew = xtrain
        conditioned = False
    if np.ndim( etrain )==0:
        if ( etrain==None )+( etrain==0 ):
            etrain = perturb*np.ones( n )
        else:
            etrain = etrain*np.ones( n )
    elif ( np.all( etrain )==None )+( np.all( etrain )==0 ):
        etrain = perturb*np.ones( n )
    elif ( np.ndim( etrain )==2 ):
        etrain = etrain.flatten()
    if np.ndim( enew )==0:
        if ( enew==None )+( enew==0 ):
            enew = perturb*np.ones( n )
        else:
            enew = enew*np.ones( n )
    elif ( np.all( enew )==None )+( np.all( enew )==0 ):
        enew = perturb*np.ones( n )
    elif ( np.ndim( enew )==2 ):
        enew = enew.flatten()

    # The number of predictive points:
    p = np.shape( xnew )[0]

    mnew = mfunc( xnew, **mpars ).flatten()
    kpp = cfunc( xnew, None, **cpars ).flatten()

    # Evaluate the predictive means and variances:
    if conditioned==True:

        n = np.shape( xtrain )[0]
        if np.ndim( etrain )==0:
            if ( etrain==None )+( etrain==0 ):
                etrain = perturb*np.ones( n )
            else:
                etrain = etrain*np.ones( n )
        elif ( np.all( etrain )==None )+( np.all( etrain )==0 ):
            etrain = perturb*np.ones( n )
        elif ( np.ndim( etrain )==2 ):
            etrain = etrain.flatten()

        # Precomputations:

        mtrain = mfunc( xtrain, **mpars )
        rtrain = np.matrix( dtrain.flatten() - mtrain.flatten() ).T
        mnew = np.matrix( mnew ).T
        kpp = np.matrix( kpp + enew**2. ).T
        Kn = np.matrix( cfunc( xtrain, xtrain, **cpars ) + np.diag( etrain**2. ) )
        Knp = np.matrix( cfunc( xtrain, xnew, **cpars ) )

        # Use Cholesky decompositions to efficiently calculate the predictive mean:
        L = np.linalg.cholesky( Kn )
        Linv_rtrain = np.matrix( scipy.linalg.lu_solve( scipy.linalg.lu_factor( L ), rtrain ) )
        LTinv_Linv_rtrain = np.linalg.lstsq( L.T, Linv_rtrain )[0]
        mu = np.array( mnew + Knp.T * LTinv_Linv_rtrain ).flatten()

        # Now do similar for the predictive variances:
        Linv_Knp = scipy.linalg.lu_solve( scipy.linalg.lu_factor( L ), Knp )
        T = np.sum( Linv_Knp**2, axis=0 )
        sig = np.sqrt( np.array( kpp ).flatten() - T.flatten() )

    else:
        
        ctrain = None
        rtrain = None
        etrain = None
        cedge = None
        mu = mfunc( xnew, **mpars ).flatten()
        sig = np.sqrt( kpp.flatten()**2. + enew**2. )

    mu = np.reshape( mu, [ p, 1 ] )
    sig = np.reshape( sig, [ p, 1 ] )
        
    return mu, sig


def logp_builtin( gp_obj, perturb=PERTURB ):
    """
    Uses the contents of the gp object to calculate its log likelihood. The
    logp() routine is actually used to perform the calculation. Note that
    the latter can be called directly if for some reason it is preferable to
    do the precomputations separately outside the routine.
    """

    xtrain = gp_obj.xtrain
    dtrain = gp_obj.dtrain
    etrain = gp_obj.etrain
    xinduc = gp_obj.xinduc
    mfunc = gp_obj.mfunc
    mpars = gp_obj.mpars
    cfunc = gp_obj.cfunc
    cpars = gp_obj.cpars
    n = np.shape( dtrain )[0]
    if mpars==None:
        mpars = {}
    if cpars==None:
        cpars = {}
    if np.ndim( etrain )==0:
        if ( etrain==None )*( etrain==0 ):
            etrain = perturb*np.ones( n )
        else:
            etrain = etrain*np.ones( n )
    elif ( np.all( etrain )==None )+( np.all( etrain )==0 ):
        etrain = perturb*np.ones( n )
    elif ( np.ndim( etrain )==2 ):
        etrain = etrain.flatten()
        
    if mfunc==None:
        mfunc = zero_mfunc
    mu = mfunc( xtrain, **mpars )
    resids = dtrain.flatten() - mu.flatten()
    resids = np.reshape( resids, [ n, 1 ] )
    Kn = cfunc( xtrain, xtrain, **cpars )
    loglikelihood = logp( resids, Kn, etrain, perturb=perturb )

    return loglikelihood


def logp_ORIGINAL( resids=None, Kn=None, sigw=None, perturb=PERTURB ):
    """
    SUMMARY
    
    Evaluates the log likelihood of residuals that are assumed to be generated by a
    gp with a specified covariance. The mean and covariance are passed directly into
    the function as inputs, to allow flexibility in how they are actually computed.
    This can be useful when repeated evaluations of logp are required (eg. likelihood
    maximisation or MCMC), as it may be possible to optimise how these precomputations
    are done outside the function.

    CALLING

    loglikelihood = logp( resids, Kn, sigw, perturb=PERTURB )

    INPUTS
    
    'resids' [Nx1 array] - residuals between the training data and the gp mean function.
    'Kn' [NxN array] - the covariance matrix between the training inputs.
    'sigw' [float] - white noise value to be incorporated into the covariance diagonal;
          if set to None or zero, it will be set to the value of the perturb variable
          for numerical stability.
    'perturb' [float] - small perturbation to be added to the covariance diagonal for
          numerical stability if the white noise errors are set to None/zero.

    OUTPUT
    
    'loglikelihood' [float] - the gp log likelihood.
    """
    t1=time.time()
    n = np.shape( resids )[0]
    if ( np.ndim( sigw )==0 ):
        if ( sigw==None )+( sigw==0.0 ):
            sigw = perturb*np.ones( n )
        else:
            sigw = sigw*np.ones( n )
    elif ( np.all( sigw )==None )+( np.all( sigw )==0 ):
        sigw = perturb*np.ones( n )
    elif ( np.ndim( sigw )==2 ):
        sigw = sigw.flatten()
    Kn = np.matrix( Kn + np.diag( sigw**2. ) )
    r = np.matrix( resids )
    # Get the log determinant of the covariance matrix:
    sign, logdet_Kn = np.linalg.slogdet( Kn ) # cpu bottleneck
    # Calculate the product inv(c)*deldm using LU factorisations:
    invKn_r = scipy.linalg.lu_solve( scipy.linalg.lu_factor( Kn ), r ) # cpu bottleneck
    rT_invKn_r = float( r.T * np.matrix( invKn_r ) )
    # Calculate the log likelihood:
    loglikelihood = - 0.5*logdet_Kn - 0.5*rT_invKn_r - 0.5*n*np.log( 2*np.pi )

    return float( loglikelihood )

def logp( resids=None, Kn=None, sigw=None, perturb=PERTURB ):
    """
    SUMMARY
    
    Evaluates the log likelihood of residuals that are assumed to be generated by a
    gp with a specified covariance. The mean and covariance are passed directly into
    the function as inputs, to allow flexibility in how they are actually computed.
    This can be useful when repeated evaluations of logp are required (eg. likelihood
    maximisation or MCMC), as it may be possible to optimise how these precomputations
    are done outside the function.

    CALLING

    loglikelihood = logp( resids, Kn, sigw, perturb=PERTURB )

    INPUTS
    
    'resids' [Nx1 array] - residuals between the training data and the gp mean function.
    'Kn' [NxN array] - the covariance matrix between the training inputs.
    'sigw' [float] - white noise value to be incorporated into the covariance diagonal;
          if set to None or zero, it will be set to the value of the perturb variable
          for numerical stability.
    'perturb' [float] - small perturbation to be added to the covariance diagonal for
          numerical stability if the white noise errors are set to None/zero.

    OUTPUT
    
    'loglikelihood' [float] - the gp log likelihood.
    """
    t1=time.time()
    n = np.shape( resids )[0]
    if ( np.ndim( sigw )==0 ):
        if ( sigw==None )+( sigw==0.0 ):
            sigw = perturb*np.ones( n )
        else:
            sigw = sigw*np.ones( n )
    elif ( np.all( sigw )==None )+( np.all( sigw )==0 ):
        sigw = perturb*np.ones( n )
    elif ( np.ndim( sigw )==2 ):
        sigw = sigw.flatten()
    Kn = np.matrix( Kn + np.diag( sigw**2. ) )
    r = np.matrix( resids )
    t1=time.time()
    chofactor = scipy.linalg.cho_factor( Kn )
    logdetK = ( 2*np.log( np.diag( chofactor[0] ) ).sum() )
    logP = -0.5*r.T*np.mat( scipy.linalg.cho_solve( chofactor, r ) ) - 0.5*logdetK - 0.5*r.size*np.log( 2*np.pi )
    t2=time.time()
    return float( logP )



def prep_fixedcov( gp_obj, perturb=PERTURB ):
    """
    Prepares a dictionary containing variables that remain unchanged in calculating
    the log likelihood when the covariance parameters are fixed. The usage of this
    routine is along the lines of:
      >> resids = data - model
      >> kwpars = gp.prep_fixedcov()
      >> logp = gp.logp_fixedcov( resids=resids, kwpars=kwpars )
    """

    Kn = gp_obj.cfunc( gp_obj.xtrain, gp_obj.xtrain, **gp_obj.cpars )
    n = np.shape( Kn )[0]
    if np.ndim( gp_obj.etrain )==0:
        if ( gp_obj.etrain==None )+( gp_obj.etrain==0 ):
            sigw = perturb*np.ones( n )
        else:
            sigw = gp_obj.etrain*np.ones( n )
    elif ( np.all( gp_obj.etrain )==None )+( np.all( gp_obj.etrain )==0 ):
        sigw = gp_obj.etrain*np.ones( n )
    elif ( np.ndim( gp_obj.etrain )==2 ):
        sigw = gp_obj.etrain.flatten()
    else:
        sigw = gp_obj.etrain
    Kn = np.matrix( Kn + np.diag( sigw**2. ) )
    sign, logdet_Kn = np.linalg.slogdet( Kn )
    lu_factor_Kn = scipy.linalg.lu_factor( Kn )
    kwpars = { 'logdet_Kn':logdet_Kn, 'lu_factor_Kn':lu_factor_Kn }

    return kwpars


def logp_fixedcov( resids=None, kwpars=None ):
    """
    Calculates the log likehood using a specific dictionary of arguments that
    are generated using the prep_fixedcov() routine. This routine is used to
    avoid re-calculating the components of the log likelihood that remain
    unchanged if the covariance parameters are fixed, which can potentially
    save time for things like type-II maximum likelihood. The usage of this
    routine is along the lines of:
      >> resids = data - model
      >> kwpars = gp.prep_fixedcov()
      >> logp = gp.logp_fixedcov( resids=resids, kwpars=kwpars )
    """

    logdet_Kn = kwpars['logdet_Kn']
    lu_factor_Kn = kwpars['lu_factor_Kn'] # advice is to never compute invKn directly

    n = np.shape( resids )[0]

    r = np.matrix( resids )
    invKn_r = scipy.linalg.lu_solve( lu_factor_Kn, r )
    rT_invKn_r = float( r.T * np.matrix( invKn_r ) )
    loglikelihood = - 0.5*logdet_Kn - 0.5*rT_invKn_r - 0.5*n*np.log( 2*np.pi )

    return float( loglikelihood )

def zero_mfunc( x, **kwargs ):
    """
    A simple zero mean function, used whenever mfunc==None in
    any of the above routines. It takes an [NxD] array as input
    and returns an [Nx1] array of zeros.
    """
    n = np.shape( x )[0]
    return np.zeros( [ n, 1 ] )
