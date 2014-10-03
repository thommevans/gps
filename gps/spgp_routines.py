import sys, os, pdb, time
import numpy as np
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt

PERTURB = 1e-4#1e-3

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
    n = np.shape( xtrain )[0]
    d = np.shape( xtrain )[1]
    if xmesh==None:
        nmesh = 1000
        xmesh_i = np.r_[ xtrain[:,mesh_dim].min() : xtrain[:,mesh_dim].max() : 1j*nmesh ]
        xmesh = np.zeros( [ nmesh, d ] )
        for i in range( d ):
            if i!=mesh_dim:
                xmesh[:,i] = np.median( xtrain[:,i] )
            else:
                xmesh[:,i] = xmesh_i
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
        # The following step can be a computation bottleneck if there are too
        # many points on the mesh:
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
        if n<1000:
            marktype = 'o'
        elif n<2000:
            marktype = '.'
        else:
            marktype = ','
        if ( np.all( etrain==0 ) )+( np.all( etrain==None ) )+( n>=2000 ):
            ax.plot( xtrain_i, dtrain, marktype, mec='k', mfc='k', zorder=zorder0 )
        else:
            errs = etrain + np.zeros( n )
            ax.errorbar( xtrain_i, dtrain, yerr=errs, fmt=marktype, mec='k', mfc='k', ecolor='k', \
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

    # Unpack the variables stored in the GP object:
    mfunc = gp_obj.mfunc
    mpars = gp_obj.mpars
    cfunc = gp_obj.cfunc
    cpars = gp_obj.cpars
    xtrain = gp_obj.xtrain
    xinduc = gp_obj.xinduc
    dtrain = gp_obj.dtrain
    etrain = gp_obj.etrain
    n = np.shape( xtrain )[0]
    m = np.shape( xinduc )[0]
    if xnew==None:
        xnew = xtrain
        conditioned = False
    p = np.shape( xnew )[0]

    # Ensure that etrain is formatted as an array
    # and any zero entries replaced with jitter:
    if np.rank( etrain )==0:
        if ( etrain==None )+( etrain==0 ):
            etrain = perturb*np.ones( n )
    else:
        ixs = ( etrain==None )
        etrain[ixs] = perturb
        ixs = ( etrain==0 )
        etrain[ixs] = perturb
    # Do the same for enew:
    if np.rank( enew )==0:
        if ( enew==None ):
            enew = np.zeros( p )
    else:
        ixs = ( enew==None )
        enew[ixs] = perturb
        ixs = ( enew==0 )
        enew[ixs] = perturb

    if mfunc==None:
        mfunc = zero_mfunc
    if mpars==None:
        mpars = {}
    if cpars==None:
        cpars = {}
    
    # Calculate the unconditioned mean and covariance values
    # at the new input locations:
    mnew = mfunc( xnew, **mpars )
    Km = cfunc( xinduc, xinduc, **cpars ) + ( perturb**2. ) * np.eye( m )
    Kmp = cfunc( xinduc, xnew, **cpars )
    Kmn = cfunc( xinduc, xtrain, **cpars )
    knn = cfunc( xtrain, None, **cpars ).flatten()
    kpp = cfunc( xnew, None, **cpars ).flatten()
    Lm = np.linalg.cholesky( Km )

    # The following lines calculate the pxp low-rank projection matrix:
    #     Qp = (Kmp^T)*(Km^-1)*(Kmp)
    Vmp = scipy.linalg.lu_solve( scipy.linalg.lu_factor( Lm ), Kmp )
    Qp = np.array( np.matrix( Vmp ).T * Vmp )

    qpp = np.diag( Qp )
    Deltap = np.diag( kpp - qpp )
    sig2Ip = ( enew**2. ) * np.eye( p )

    # If we are using the unconditioned GP, we are finished:
    if conditioned==False:

        mu = np.array( mnew.flatten() )
        cov = np.array( Qp + Deltap + sig2Ip )

    # If we want to use the conditioned GP, we still have work to do:
    else:
        
        mtrain = mfunc( xtrain, **mpars )
        resids = dtrain.flatten() - mtrain.flatten()

        # The following lines calculate the diagonal of the nxn Gamma matrix,
        # as given by Eq C.1. To do this, we make use of the Cholesky identity
        # given by Eq B.8. Note that:
        #     sig2*Gamma = Deltan + sig2*I
        # where Deltan is the NxN diagonal matrix used in Eq 2.12.
        Lm = np.linalg.cholesky( Km )
        Vmn = scipy.linalg.lu_solve( scipy.linalg.lu_factor( Lm ), Kmn )
        gnn = 1. + ( knn.flatten() - np.sum( Vmn**2., axis=0 ).flatten() ) / ( etrain**2. )
        
        # To make things more concise, we will divide the rows of the Vmn and
        # resids arrays by the square root of the corresponding entries on the
        # Gamma matrix diagonal.
        #      Vmn         -->   Vmn * (Gamma^-0.5)
        #      resids      -->   (Gamma^-0.5) * resids
        Vmn = np.matrix( Vmn / np.tile( np.sqrt( gnn ).flatten(), [ m, 1 ] ) )
        resids = resids.flatten() / np.sqrt( gnn.flatten() )
        resids = np.matrix( np.reshape( resids, [ n, 1 ] ) )
        Vmn_resids = np.array( Vmn * resids )

        # Now we need to calculate the term involving B^-1 in Eq 2.12, which
        # we do using two Cholesky decompositions:
        W = np.array( np.linalg.cholesky( ( enew**2. ) * np.eye( m ) + np.array( Vmn*Vmn.T ) ) )
        Y = scipy.linalg.lu_solve( scipy.linalg.lu_factor( W ), Vmn_resids )
        H = np.linalg.lstsq( Lm, Kmp )[0]
        J = scipy.linalg.lu_solve( scipy.linalg.lu_factor( W ), H )

        # Finally, we use Eqs 2.9 and 2.12 to calculate the predictive mean and
        # covariance matrix of the GP:
        mu = np.array( mnew.flatten() + np.array( np.matrix( J ).T * np.matrix( Y ) ).flatten() )
        KmpTBinvKmp = ( enew**2. ) * np.array( np.matrix( J ).T * np.matrix( J ) )
        cov = np.array( Deltap + sig2Ip + KmpTBinvKmp )

    mu = np.reshape( mu, [ p, 1 ] )
    cov = np.reshape( cov, [ p, p ] )

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

    # Unpack the variables stored in the GP object:
    mfunc = gp_obj.mfunc
    mpars = gp_obj.mpars
    cfunc = gp_obj.cfunc
    cpars = gp_obj.cpars
    xtrain = gp_obj.xtrain
    xinduc = gp_obj.xinduc
    dtrain = gp_obj.dtrain
    etrain = gp_obj.etrain
    n = np.shape( xtrain )[0]
    m = np.shape( xinduc )[0]
    p = np.shape( xnew )[0]
    if mfunc==None:
        mfunc = zero_mfunc
    if mpars==None:
        mpars = {}
    if cpars==None:
        cpars = {}
    if xnew==None:
        xnew = xtrain
        conditioned = False
    # Ensure that etrain is formatted as an array
    # and any zero entries replaced with jitter:
    if np.rank( etrain )==0:
        if ( etrain==None )+( etrain==0 ):
            etrain = perturb*np.ones( n )
    else:
        ixs = ( etrain==None )
        etrain[ixs] = perturb
        ixs = ( etrain==0 )
        etrain[ixs] = perturb
    # Do the same for enew:
    if np.rank( enew )==0:
        if ( enew==None ):
            enew = np.zeros( p )
    else:
        ixs = ( enew==None )
        enew[ixs] = perturb
        ixs = ( enew==0 )
        enew[ixs] = perturb

    # Calculate the unconditioned mean and covariance values
    # at the new input locations:
    mnew = mfunc( xnew, **mpars )
    kpp = cfunc( xnew, None, **cpars ).flatten()

    # If we are using the unconditioned GP, we are finished:

    if conditioned==False:

        mu = mnew.flatten()
        sig = np.sqrt( kpp.flatten() + ( enew**2. ) )

    # If we want to use the conditioned GP, we still have work to do:
    else:
        
        mtrain = mfunc( xtrain, **mpars )
        Km = cfunc( xinduc, xinduc, **cpars ) + ( perturb**2. ) * np.eye( m )
        Kmn = cfunc( xinduc, xtrain, **cpars )
        Kmp = cfunc( xinduc, xnew, **cpars )
        knn = cfunc( xtrain, None, **cpars ).flatten()
        resids = dtrain.flatten() - mtrain.flatten()

        # The following lines calculate the diagonal of the NxN Gamma matrix,
        # as given by Eq C.1. To do this, we make use of the Cholesky identity
        # given by Eq B.8. Note that:
        #     sig2*Gamma = Delta + sig2*I
        # where Delta is the diagonal matrix used in Eq 2.12.
        Lm = np.linalg.cholesky( Km )
        Vmn = scipy.linalg.lu_solve( scipy.linalg.lu_factor( Lm ), Kmn )

        # Diagonal of QN:
        Qnn_diag = np.sum( Vmn**2., axis=0 ).flatten()

        # Diagonal of the D=sig2*Gamma matrix:
        D_diag = knn - Qnn_diag + etrain**2.

        # To make things more concise, we will divide the rows of the Vmn and
        # resids arrays by the square root of the corresponding entries on the
        # Gamma matrix diagonal. 
        #      Vmn         -->   Vmn * (Gamma^-0.5)
        #      resids      -->   (Gamma^-0.5) * resids
        Vmn = np.matrix( Vmn / np.tile( np.sqrt( D_diag ).flatten(), [ m, 1 ] ) )
        resids = resids.flatten() / np.sqrt( D_diag.flatten() )
        resids = np.matrix( np.reshape( resids, [ n, 1 ] ) )
        Vmn_resids = np.array( Vmn * resids )

        # Now we need to calculate the terms involving B^-1 in Eq 2.12, which
        # we do using two Cholesky decompositions:
        W = np.array( np.linalg.cholesky( np.eye( m ) + np.array( Vmn*Vmn.T ) ) )
        Y = scipy.linalg.lu_solve( scipy.linalg.lu_factor( W ), Vmn_resids )
        H = np.linalg.lstsq( Lm, Kmp )[0]
        J = scipy.linalg.lu_solve( scipy.linalg.lu_factor( W ), H )

        # Finally, we use Eq 2.12 to calculate the predictive mean and standard
        # deviation of the GP:
        mu = mnew.flatten() + np.array( np.matrix( J ).T * np.matrix( Y ) ).flatten()
        sig = np.sqrt( kpp.flatten() + ( enew**2. ) \
                       - np.sum( H**2., axis=0 ).flatten() \
                       + np.sum( J**2., axis=0 ).flatten() )
        # Note that:
        #   np.sum( H**2., axis=0 ) = diagonal of (H^T)*H
        #   np.sum( J**2., axis=0 ) = diagonal of (J^T)*J

    mu = np.reshape( mu, [ p, 1 ] )
    sig = np.reshape( sig, [ p, 1 ] )
    return mu, sig


def logp_builtin( gp_obj, perturb=None ):
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
    m = np.shape( xinduc )[0]
    if mpars==None:
        mpars = {}
    if cpars==None:
        cpars = {}
    # Ensure that etrain is formatted as an array
    # and any zero entries replaced with jitter:
    if np.rank( etrain )==0:
        if ( etrain==None )+( etrain==0 ):
            etrain = perturb*np.ones( n )
    else:
        ixs = ( etrain==None )
        etrain[ixs] = perturb
        ixs = ( etrain==0 )
        etrain[ixs] = perturb
    if mfunc==None:
        mfunc = zero_mfunc
    mu = mfunc( xtrain, **mpars )
    resids = dtrain.flatten() - mu.flatten()
    resids = np.reshape( resids, [ n, 1 ] )
    if xinduc==None:
        print 'Must specify inducing inputs (xinduc)'
        pdb.set_trace()
    Km = cfunc( xinduc, xinduc, **cpars )
    Kmn = cfunc( xinduc, xtrain, **cpars )            
    knn = cfunc( xtrain, None, **cpars )
    loglikelihood = logp( resids, Km, Kmn, knn, etrain, perturb=perturb )

    return loglikelihood



def logp( resids=None, Km=None, Kmn=None, knn=None, sigw=None, perturb=PERTURB ):
    """
    SUMMARY
    
    Evaluates the log likelihood of residuals that are assumed to be generated by a
    gp with a specified covariance. The mean and covariance are passed directly into
    the function as inputs, to allow flexibility in how they are actually computed.
    This can be useful when repeated evaluations of logp are required (eg. likelihood
    maximisation or MCMC), as it may be possible to optimise how these precomputations
    are done outside the function.

    The loglikelihood is calculated according to:

        loglikelihood = -0.5*n*np.log( 2*np.pi ) - 0.5*L1 - 0.5*L2

    where 'n' is the number of data points and:

        L1 = logdet[ (Kmm^-1)*( Kmm+Kmn*(W^-1)*(Kmn^T) ) ] - logdet(W)

        L2 = norm[ V*r ]^2 - norm[ (U^-1)*Kmn*(W^-1)*r ]^2

        W = diag[ Knn - (Kmn^T)*(Km^-1)*Kmn ] + (sigw^2)*I

        V*(V^T) = W

        U*(U^T) = (Kmn^T)*(Km^-1)*Kmn + W


    CALLING

    loglikelihood = logp( resids, Kn, sigw, perturb=PERTURB )

    INPUTS
    
    'resids' [Nx1 array] - residuals between the training data and the gp mean function.
    'Kn' [NxN array] - the covariance matrix between the training inputs.
    'sigw' [Nx1 array or float] - white noise value to be incorporated into the covariance diagonal;
          if set to None or zero, it will be set to the value of the perturb variable
          for numerical stability.
    'perturb' [float] - small perturbation to be added to the covariance diagonal for
          numerical stability if the white noise errors are set to None/zero.

    OUTPUT
    
    'loglikelihood' [float] - the gp log likelihood.
    """

    # Convert sigw to an array and replace any zero
    # entries with jitter:
    if np.rank( sigw )==0:
        if ( sigw==None )+( sigw==0 ):
            sigw = perturb*np.ones( n )
    else:
        ixs = ( sigw==None )
        sigw[ixs] = perturb
        ixs = ( sigw==0 )
        sigw[ixs] = perturb

    # Unpack and prepare:
    n = np.shape( Kmn )[1] # number of data points
    m = np.shape( Kmn )[0] # number of inducing variables
    Km = np.matrix( Km + ( perturb**2. ) * np.eye( m ) )
    Kmn = np.matrix( Kmn )
    knn = ( knn + perturb**2. ).flatten()
    r = np.reshape( resids, [ n, 1 ] )
    Sig2_diag = sigw**2.

    # Calculate the diagonal entries of the Qnn matrix, where:
    #    Qnn = (Kmn^T)*(Kmm^-1)*Kmn
    H = np.linalg.cholesky( Km )
    V = np.array( scipy.linalg.lu_solve( scipy.linalg.lu_factor( H ), Kmn ) )
    Qnn_diag = np.sum( V**2., axis=0 )

    # Generate an array holding the diagonal entries of the D matrix, where:
    #    D = Qnn + diag[ Knn - Qnn ]
    D_diag = ( knn - Qnn_diag + Sig2_diag ).flatten()

    # Convert V to V*(D^-0.5) and compute V*(D^-1)*V:
    V = np.matrix( V/np.tile( np.sqrt( D_diag ), [ m, 1 ] ) )
    VVT = V*V.T

    # Convert r to (D^-0.5)*r and compute (r^T)*(D^-1)*r:
    r = np.matrix( np.reshape( r.flatten()/np.sqrt( D_diag ), [ n, 1 ] ) )

    # To obtain L1, compute:
    #        L1 = 0.5*logdet(B) + 0.5*logdet(D)
    # where:
    #   B*(B^T) = I + V*(V^T)
    #           = I + (H^-1)*Kmn*(D^-1)*(Kmn^T)*(H^-T)
    #           = (H^-1)*[ Kmm + Kmn*(D^-1)*(Kmn^T) ]*(H^-T)
    #           = (H^-1)*[ Kmm + Kmn*(D^-1)*(Kmn^T) ]*(Km^-1)*H
    #   det[ (H^-1)*[ Kmm + Kmn*(D^-1)*(Kmn^T) ]*(Km^-1)*H ] = prod[ diag(B)^2 ]
    #   (this is a standard result of the Cholesky decomposition)
    #   --> logdet[ ( Kmm + Kmn*(D^-1)*(Kmn^T) )*(Km^-1) ] = 2*sum[ diag(B) ]
    #   (using standard results det[ X*Y ]=det[X]*det[Y] and det[X^-1]=1/det[X])
    B = np.linalg.cholesky( np.matrix( np.eye( m ) ) + VVT )
    logdetB = 2*np.sum( np.log( np.diag( B ) ) )
    logdetD = np.sum( np.log( D_diag ) )
    L1 = 0.5*( logdetB + logdetD )

    # To obtain L2, compute:
    #        L2 = 0.5*(r^T)*r - 0.5*(Y^T)*Y
    # where:
    #   (Y^T)*Y = (r^T)*(D^-0.5)*(Z^T)*Z*(D^0.5)*r
    #         Z = (B^-1)*V*(D^-0.5)
    #           = (B^-1)*(H^-1)*Kmn*(D^-0.5)
    #           = (B^-1)*(H^-1)*Kmn*(D^-0.5)
    #       Z^T = (D^-0.5)*(Kmn^T)*(H^-T)*(B^-T)
    # so that:
    #   (Y^T)*Y = (r^T)*(D^-1)*(Kmn^T)*(H^-T)*(B^-T)*(B^-1)*(H^-1)*Kmn*(D^-1)*r
    #           = norm[ H*B*Kmn*(D^-1)*r ]^2
    # as it can be verified that:
    #   (H*B)*[(H*B)^T] = Kmm + Kmn*(D^-1)*(Kmn^T)
    # so that:
    #   (H^-T)*(B^-T)*(B^-1)*(H^-1) = (Kmm + Kmn*(D^-1)*(Kmn^T))^-1
    rTr = float( r.T*r )
    Z = np.matrix( scipy.linalg.lu_solve( scipy.linalg.lu_factor( B ), V ) )
    Y = Z*r
    YTY = float( Y.T*Y )
    L2 = 0.5*( rTr - YTY )
    L3 = 0.5*n*np.log( 2*np.pi )

    return -float(  L1 + L2 + L3 )


def prep_fixedcov( gp_obj, perturb=PERTURB ):
    """
    Prepares a dictionary containing variables that remain unchanged in calculating
    the log likelihood when the covariance parameters are fixed. The usage of this
    routine is along the lines of:
      >> resids = data - model
      >> kwpars = gp.prep_fixedcov()
      >> logp = gp.logp_fixedcov( resids=resids, kwpars=kwpars )
    """

    # Unpack the variables stored in the GP object:
    mfunc = gp_obj.mfunc
    mpars = gp_obj.mpars
    cfunc = gp_obj.cfunc
    cpars = gp_obj.cpars
    xtrain = gp_obj.xtrain
    xinduc = gp_obj.xinduc
    dtrain = gp_obj.dtrain
    sigw = gp_obj.etrain

    Kmn = cfunc( xinduc, xtrain, **cpars )
    n = np.shape( Kmn )[1] # number of data points
    m = np.shape( Kmn )[0] # number of inducing variables
    Km = cfunc( xinduc, xinduc, **cpars ) + ( perturb**2. ) * np.eye( m )
    knn = cfunc( xtrain, None, **cpars ).flatten()
    knn = ( knn + perturb**2. ).flatten()

    # Convert sigw to an array and replace any zero
    # entries with jitter:
    if np.rank( sigw )==0:
        if ( sigw==None )+( sigw==0 ):
            sigw = perturb*np.ones( n )
    else:
        ixs = ( sigw==None )
        sigw[ixs] = perturb
        ixs = ( sigw==0 )
        sigw[ixs] = perturb
    Sig2_diag = sigw**2.

    # Calculate the diagonal entries of the Qnn matrix, where:
    #    Qnn = (Kmn^T)*(Kmm^-1)*Kmn
    H = np.linalg.cholesky( Km )
    V = np.array( scipy.linalg.lu_solve( scipy.linalg.lu_factor( H ), Kmn ) )
    Qnn_diag = np.sum( V**2., axis=0 )

    # Generate an array holding the diagonal entries of the D matrix, where:
    #    D = Qnn + diag[ Knn - Qnn ]
    D_diag = ( knn - Qnn_diag + Sig2_diag ).flatten()

    # CHECK THIS IS DOING THE RIGHT THING:
    # Convert V to V*(D^-0.5) and compute V*(D^-1)*V: 
    V = np.matrix( V/np.tile( np.sqrt( D_diag ), [ m, 1 ] ) )
    VVT = V*V.T

    # To obtain L1, compute:
    #        L1 = 0.5*logdet(B) + 0.5*logdet(D)
    # where:
    #   B*(B^T) = I + V*(V^T)
    #           = I + (H^-1)*Kmn*(D^-1)*(Kmn^T)*(H^-T)
    #           = (H^-1)*[ Kmm + Kmn*(D^-1)*(Kmn^T) ]*(H^-T)
    #           = (H^-1)*[ Kmm + Kmn*(D^-1)*(Kmn^T) ]*(Km^-1)*H
    #   det[ (H^-1)*[ Kmm + Kmn*(D^-1)*(Kmn^T) ]*(Km^-1)*H ] = prod[ diag(B)^2 ]
    #   (the above is a standard result of the Cholesky decomposition)
    #   --> logdet[ ( Kmm + Kmn*(D^-1)*(Kmn^T) )*(Km^-1) ] = 2*sum[ diag(B) ]
    #   (using standard results det[ X*Y ]=det[X]*det[Y] and det[X^-1]=1/det[X])
    B = np.linalg.cholesky( np.matrix( np.eye( m ) ) + VVT )
    logdetB = 2*np.sum( np.log( np.diag( B ) ) )
    logdetD = np.sum( np.log( D_diag ) )

    L1 = 0.5*( logdetB + logdetD )
    Z = np.matrix( scipy.linalg.lu_solve( scipy.linalg.lu_factor( B ), V ) )
    L3 = 0.5*n*np.log( 2*np.pi )
    sqrt_D_diag = np.reshape( np.sqrt( D_diag ), [ n, 1 ] )

    kwpars = { 'L1':L1, 'L3':L3, 'Z':Z, 'sqrt_D_diag':sqrt_D_diag }

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

    L1 = kwpars['L1']
    L3 = kwpars['L3']
    Z = kwpars['Z']
    sqrt_D_diag = kwpars['sqrt_D_diag']
    r = np.matrix( resids/sqrt_D_diag )
    
    # rTr should be rT*(D^(-1))*r
    rTr = float( r.T*r )
    Y = Z*r
    YTY = float( Y.T*Y )
    L2 = 0.5*( rTr - YTY )

    return -float( L1 + L2 + L3 )


def prep_fixedcov_OLD( gp_obj, perturb=PERTURB ):
    """
    Prepares a dictionary containing variables that remain unchanged in calculating
    the log likelihood when the covariance parameters are fixed. The usage of this
    routine is along the lines of:
      >> resids = data - model
      >> kwpars = gp.prep_fixedcov()
      >> logp = gp.logp_fixedcov( resids=resids, kwpars=kwpars )
    """

    # Ensure that etrain is formatted as an array
    # and any zero entries replaced with jitter:
    etrain = gp_obj.etrain
    if np.rank( etrain )==0:
        if ( etrain==None )+( etrain==0 ):
            etrain = perturb*np.ones( n )
    else:
        ixs = ( etrain==None )
        etrain[ixs] = perturb
        ixs = ( etrain==0 )
        etrain[ixs] = perturb
    # Do the same for enew:
    if np.rank( enew )==0:
        if ( enew==None ):
            enew = np.zeros( p )
    else:
        ixs = ( enew==None )
        enew[ixs] = perturb
        ixs = ( enew==0 )
        enew[ixs] = perturb

    Km = gp_obj.cfunc( gp_obj.xinduc, gp_obj.xinduc, **gp_obj.cpars )
    Kmn = gp_obj.cfunc( gp_obj.xinduc, gp_obj.xtrain, **gp_obj.cpars )
    knn = gp_obj.cfunc( gp_obj.xtrain, None, **gp_obj.cpars )
    n = np.shape( Kmn )[1] 
    m = np.shape( Kmn )[0]

    Km = np.matrix( Km + ( perturb**2. ) * np.eye( m ) )
    Kmn = np.matrix( Kmn )
    knn = np.matrix( knn + perturb**2. )
    
    L = np.linalg.cholesky( Km )
    Vmn = np.matrix( scipy.linalg.lu_solve( scipy.linalg.lu_factor( L ), Kmn ) )

    gnn = 1. + ( knn.flatten() - np.sum( np.power( Vmn, 2. ), axis=0 ) ) / ( etrain**2. )
    gnn = np.reshape( gnn, [ n, 1 ] )
    
    Vmn = Vmn / np.tile( np.sqrt( gnn ).T, [ m, 1 ] )
    VmnVmnT = Vmn * Vmn.T
    W = np.linalg.cholesky( np.matrix( ( etrain**2. ) * np.eye( m ) ) + VmnVmnT )
    Z = scipy.linalg.lu_solve( scipy.linalg.lu_factor( W ), Vmn )
    Z = np.matrix( Z )
    L1 = 0.5 * ( 2 * np.sum( np.log( np.diag( W ) ) ) + np.sum( np.log( gnn ) ) \
         + ( n-m ) * np.log( gp_obj.etrain**2. ) )
    L3 = 0.5*n*np.log( 2*np.pi )
    kwpars = { 'L1':L1, 'L3':L3, 'gnn':gnn, 'Z':Z, 'sigw':etrain }
    
    return kwpars

    

def zero_mfunc( x, **kwargs ):
    """
    A simple zero mean function, used whenever mfunc==None in
    any of the above routines. It takes an [NxD] array as input
    and returns an [Nx1] array of zeros.
    """
    n = np.shape( x )[0]
    return np.zeros( [ n, 1 ] )
