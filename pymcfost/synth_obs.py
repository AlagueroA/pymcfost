import copy

import astropy.io.fits as fits
from astropy.convolution import convolve, convolve_fft
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

from vip_hci.psfsub import median_sub, pca
from vip_hci.preproc import frame_rotate
from vip_hci.metrics import snrmap

from .utils import *
from .image import *


def apply_atm_bg(I, sky_mag, wavelength, pixel_scale) :
    '''
    Apply a constant sky background on the image (2D-array) I.

    I : ndarray
        A single 2d-array
    sky_mag : float
        Sky magnitude/arcsec²
    wavelength : float
        Wavelength in µm
    pixel_scale : float
        pixel scale in arcsec
    '''


    #sky intensity
    F_0 = ref_flux(wavelength)
    I_0 = F_0 / (len(I)*len(I[0])*pixel_scale**2)           #intensity in W.m-2.arcsec-2

    I_sky_arcsec2 = I_0*10**(-sky_mag/2.5)             #in W.m-2.arcsec-2
    I_sky = I_sky_arcsec2 * pixel_scale**2         #in W.m-2.px-1, unit of I

    #creation of the array and addition to intensity
    I_sky = I_sky*np.ones((len(I),len(I[0])))

    I = I + I_sky



def apply_photon_noise(I, wavelength, pixel_scale, telescope_surface, exp_time) :
    '''
    Applies photon noise on an image I in W.m⁻².px⁻¹.

    #args
    I : ndarray
        A single 2d-array in W.m⁻².px⁻¹
    wavelength : float
        Wavelength in µm, needed to convert to a photon number
    pixel_scale : float
        pixel scale in arcsec
    telescope_surface : float
        The telescope surface in m²
    exp_time : float
        Exposure time in s

    #return
    The array with photon noise applied 

    '''

    #conversion of I to a photon number
    freq = sc.c / (wavelength*1e-6)
    N_ph = Wm2_to_ph(I, nu=freq, surface=telescope_surface, exp_time=exp_time, pixelscale=pixel_scale)

    #generation of the noise
    N_noise = np.zeros((len(N_ph),len(N_ph[0])))
    for i in range(len(N_ph)):
        for j in range(len(N_ph[0])):
            coeff = random.random()
            sign = random.random()
            if sign >= 0.5 :
                N_noise[i][j] = np.sqrt(N_ph[i][j])*coeff
            else :
                N_noise[i][j] = -np.sqrt(N_ph[i][j])*coeff
                if np.abs(N_noise[i][j]) > N_ph[i][j] :   #if noise is over the signal, put the signal to 0
                    N_noise[i][j] = - N_ph[i][j]

    #Addition of the noise
    N_ph = N_ph + N_noise
    I = ph_to_Wm2(N_ph, nu=freq, surface=telescope_surface, exp_time=exp_time, pixelscale=pixel_scale)

    return(I)


def apply_readout_noise(I, wavelength, pixel_scale, telescope_surface, exp_time, RON=2) :
    '''
    Applies readout noise on an image I in W.m⁻².px⁻¹.

    #args
    I : ndarray
        A single 2d-array in W.m⁻².px⁻¹
    wavelength : float
        Wavelength in µm, needed to convert to a photon number
    pixel_scale : float
        pixel scale in arcsec
    telescope_surface : float
        The telescope surface in m²
    exp_time : float
        Exposure time in s
    RON : int
        Readout noise in e/pix

    #return
    The array with readout noise applied

    '''

    #conversion of I to a photon number
    freq = sc.c / (wavelength*1e-6)
    N_ph = Wm2_to_ph(I, nu=freq, surface=telescope_surface, exp_time=exp_time, pixelscale=pixel_scale)

    #generation of the noise
    N_noise = np.zeros((len(N_ph),len(N_ph[0])))
    for i in range(len(N_ph)):
        for j in range(len(N_ph[0])):
            coeff = random.random()
            sign = random.random()
            if sign >= 0.5 :
                N_noise[i][j] = RON*coeff
            else :
                N_noise[i][j] = -RON*coeff
                if np.abs(N_noise[i][j]) > N_ph[i][j] :   #if noise is over the signal, put the signal to 0
                    N_noise[i][j] = - N_ph[i][j]

    #Addition of the noise
    N_ph = N_ph + N_noise
    I = ph_to_Wm2(N_ph, nu=freq, surface=telescope_surface, exp_time=exp_time, pixelscale=pixel_scale)

    return(I)


def convolution(I, pixelscale, conv_kernel, coro_diff, coronagraph=None, adi_prep=False, angle_list=None):
    '''
    Take a 2d-single array, mask the central parts of it, convolve it with a PSF sequence and finally add the coronagraph diffraction.

    #args
    I : ndarray
        A single 2d-array
    pixelscale : float
        Image pixel scale in arcsec
    conv_kernel : 2d-array or list
        The PSF sequence. Can be passed as a 2d array if there is only one psf
    coro_diff : 2d-array or list
        Coronagraph diffraction (on-axis PSF) sequence. Can be passed as a 2d array if there is only one psf. If None, must be passed as an empty list.
    coronagraph : float or None
        If not None, apply a central opaque mask of radius to be given in mas
    adi_prep : bool, optional
        Prepare the cube for ADI by rotating the MCFOST image before convolving.
    angle_list : list or 1d-array
        [only used if adi_prep is True]
        Parallactic angle list for the image to be rotated

    #return
    The convolved sequence
    '''

    #
    print('Convolution...')

    nx = len(I)
    ny = len(I[0])

    if conv_kernel == []:       #if no convolution
        conv_kernel=None
        I_list = [I]

    if conv_kernel is not None :
        if not isinstance(conv_kernel, list):           #if psf is a single 2d-array, making it a list
            conv_kernel = [conv_kernel]
        if coro_diff==[] :                              #if no coronagraph diffraction, initializing an array of 0
            coro_diff = np.zeros((nx, ny))
        if not isinstance(coro_diff, list):             #if coro. diffraction is a single 2d-array, making it a list of the same length of the psf list
            coro_diff_temp = coro_diff
            coro_diff = []
            for i_ in range(len(conv_kernel)):
                coro_diff.append(coro_diff_temp)
        if len(conv_kernel) != len(coro_diff) :
            raise ValueError('PSF list musts be the same size of the coronagraph diffraction list')

        #Lists that will contain the convolved images
        I_list = []

        #Assuming the intensity of the central star is the one of the brightest pixel
        I_star = np.max(I)

        #Central mask
        if coronagraph is not None :

            posx = np.linspace(-nx/2, nx/2, nx)
            posy = np.linspace(-ny/2, ny/2, ny)
            meshx, meshy = np.meshgrid(posx, posy)
            radius_pixel = np.sqrt(meshx ** 2 + meshy ** 2)
            radius_as = radius_pixel * pixelscale

            I[radius_as < coronagraph] = 0.0

        #Convolving image and adding diffraction for each psf of the list
        compt = 0
        I_mem = copy.deepcopy(I)   #keeping MCFOST image in memory inc ase of rotation
        for diff, psf in zip(coro_diff, conv_kernel) :

            #rotation for further ADI
            if adi_prep :
                if compt == 0 :
                    print('... and rotation')
                I = frame_rotate(I_mem, angle_list[compt])
                compt += 1

            #convolution with the psf
            I_temp = convolve_fft(I, psf)

            #construction of the coronagraphic image
            I_coro = I_star * diff
            I_temp += I_coro

            #put the result in the science image cube
            I_list.append(I_temp)

    return(I_list)



def recombine(I_list, mode='median'):
    '''
    Recombine an image sequence into a single image.

    #args
    I_list : list-like of 2d-arrays
        The image sequence
    mode : str, optional
        The recombination method in 'median' or 'average' or 'mean'

    #return
    The array that results the recombination
    '''

    #
    I_list = np.array(I_list)
    
    if len(I_list.shape) == 2 :    #if the cube is a single image
        I = I_list
        print('The cube is a single image, no need of collapsing.')
        
    elif mode.lower() == 'mean' or mode.lower() == 'average' :
        print('Collapsing the cube. Mode '+mode.lower())
        I = np.zeros((len(I_list[0]), len(I_list[0][0])))
        for ind_x in range(len(I_list[0])):
             for ind_y in range(len(I_list[0][0])):
                 for ind_I in range(len(I_list)):
                    I[ind_x][ind_y] += I_list[ind_I][ind_x][ind_y] / len(I_list)

    elif mode.lower() == 'median' :
        print('Collapsing the cube. Mode '+mode.lower())
        I = np.zeros((len(I_list[0]), len(I_list[0][0])))
        for ind_x in range(len(I_list[0])):
             for ind_y in range(len(I_list[0][0])):
                pix_list_I = []
                for ind_I in range(len(I_list)):
                    pix_list_I.append(I_list[ind_I][ind_x][ind_y])
                I[ind_x][ind_y] = np.median(pix_list_I)

        I = np.median(I_list, axis=0)

    else:
        raise AttributeError('please select a valid combination method : None, mean/average, median')

    return(I)


def RDI(I_list, I_ref_list):
    '''
    Subtract a reference sequence from an image sequence and returns the result. Both have to be already convolved. (RDI ready data)

    #args
    I_list : 2d-array or list
        The image sequence as a list or a 2d-array if it is a single image.
    I_ref_list : 2d-array or list
        The reference sequence. Must be the same shape as I_list.

    #return
    The RDI sequence
    '''

    #
    print('applying RDI')

    #checks
    if not isinstance(I_list, list):
        I_list = [I_list]
    if not isinstance(I_ref_list, list):
        I_ref_list = [I_ref_list]
    if len(I_list) != len(I_ref_list) :
        raise ValueError('Sequences must have the same size ! Here image sequence has '+str(len(I_list))+' exposures while reference sequence has '+str(len(I_ref_list)))
    if I_list[0].shape != I_ref_list[0].shape :
        raise ValueError('2d-arrays of the sequences must have the same size ! Here images are '+str(I_list.shape)+' while references are '+str(I_ref_list.shape))

    #reference subtraction
    I_RDI_list = []
    for ind_I in range(len(I_list)):
        I_RDI_list.append(I_list[ind_I]-I_ref_list[ind_I])

    return(I_RDI_list)


def mADI(**kwargs):
    '''
    Process an image sequence using median Angular Differential Imaging using vip_hci.psfsub.medsub.median_sub .

    **kwargs are keyword arguments of vip_hci.psfsub.medsub.median_sub :


    Parameters
    ----------
    cube : numpy ndarray, 3d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    scale_list : numpy ndarray, 1d, optional
        If provided, triggers mSDI reduction. These should be the scaling
        factors used to re-scale the spectral channels and align the speckles
        in case of IFS data (ADI+mSDI cube). Usually, these can be approximated
        by the last channel wavelength divided by the other wavelengths in the
        cube (more thorough approaches can be used to get the scaling factors,
        e.g. with ``vip_hci.preproc.find_scal_vector``).
    flux_sc_list : numpy ndarray, 1d
        In the case of IFS data (ADI+SDI), this is the list of flux scaling
        factors applied to each spectral frame after geometrical rescaling.
        These should be set to either the ratio of stellar fluxes between the
        last spectral channel and the other channels, or to the second output
        of `preproc.find_scal_vector` (when using 2 free parameters). If not
        provided, the algorithm will still work, but with a lower efficiency
        at subtracting the stellar halo.
    fwhm : float or 1d numpy array
        Known size of the FHWM in pixels to be used. Default is 4.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame).
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    mode : {'fullfr', 'annular'}, str optional
        In ``fullfr`` mode only the median frame is subtracted, in ``annular``
        mode also the 4 closest frames given a PA threshold (annulus-wise) are
        subtracted.
    nframes : int or None, optional
        Number of frames (even value) to be used for building the optimized
        reference PSF when working in ``annular`` mode. None by default, which
        means that all frames, excluding the thresholded ones, are used.
    sdi_only: bool, optional
        In the case of IFS data (ADI+SDI), whether to perform median-SDI, or
        median-ASDI (default).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "border_mode", "mask_val",
        "edge_blend", "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``)
    Returns
    -------
    cube_out : numpy ndarray, 3d
        [full_output=True] The cube of residuals.
    cube_der : numpy ndarray, 3d
        [full_output=True] The derotated cube of residuals.
    frame : numpy ndarray, 2d
        Median combination of the de-rotated cube.

    '''

    print('Using median ADI from vip_hci :')

    #checks : full_output mandatory, converting lists into arrays
    kwargs['full_output'] = True
    if isinstance(kwargs['cube'], list):
        kwargs['cube'] = np.array(kwargs['cube'])
    if isinstance(kwargs['angle_list'], list):
        kwargs['angle_list'] = np.array(kwargs['angle_list'])

    #return the derotated cube
    I_list = median_sub(**kwargs)[1]

    return(I_list)


def PCA(**kwargs):
    '''
    Process an image sequence using median Angular Differential Imaging using vip_hci.psfsub.pca_fullfr.pca .

    **kwargs are keyword arguments of vip_hci.psfsub.pca_fullfr.pca :

       Parameters
    ----------
    cube : str or numpy ndarray, 3d or 4d
        Input cube (ADI or ADI+mSDI). If 4D, the first dimension should be
        spectral. If a string is given, it must correspond to the path to the
        fits file to be opened in memmap mode (incremental PCA-ADI of 3D cubes
        only).
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : 3d or 4d numpy ndarray, or list of 3D numpy ndarray, optional
        Reference library cube for Reference Star Differential Imaging. Should
        be 3D, except if input cube is 4D and no scale_list is provided,
        reference cube can then either be 4D or a list of 3D cubes (i.e.
        providing the reference cube for each individual spectral cube).
    scale_list : numpy ndarray, 1d, optional
        If provided, triggers mSDI reduction. These should be the scaling
        factors used to re-scale the spectral channels and align the speckles
        in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the last channel wavelength divided by the
        other wavelengths in the cube (more thorough approaches can be used
        to get the scaling factors, e.g. with
        ``vip_hci.preproc.find_scal_vector``).
    ncomp : int, float or tuple of int/None, or list, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.
        * ADI (``cube`` is a 3d array): if an int is provided, ``ncomp`` is the
          number of PCs extracted from ``cube`` itself. If ``ncomp`` is a float
          in the interval [0, 1] then it corresponds to the desired cumulative
          explained variance ratio (the corresponding number of components is
          estimated). If ``ncomp`` is a tuple of two integers, then it
          corresponds to an interval of PCs in which final residual frames are
          computed (optionally, if a tuple of 3 integers is passed, the third
          value is the step). When ``source_xy`` is not None, then the S/Ns
          (mean value in a 1xFWHM circular aperture) of the given
          (X,Y) coordinates are computed.
        * ADI+RDI (``cube`` and ``cube_ref`` are 3d arrays): ``ncomp`` is the
          number of PCs obtained from ``cube_ref``. If ``ncomp`` is a tuple,
          then it corresponds to an interval of PCs (obtained from ``cube_ref``)
          in which final residual frames are computed. If ``source_xy`` is not
          None, then the S/Ns (mean value in a 1xFWHM circular aperture) of the
          given (X,Y) coordinates are computed.
        * ADI or ADI+RDI (``cube`` is a 4d array): same input format allowed as
          above. If ``ncomp`` is a list with the same length as the number of
          channels, each element of the list will be used as ``ncomp`` value
          (be it int, float or tuple) for each spectral channel. If not a
          list, the same value of ``ncomp`` will be used for all spectral
          channels (be it int, float or tuple).
        * ADI+mSDI (``cube`` is a 4d array and ``adimsdi="single"``): ``ncomp``
          is the number of PCs obtained from the whole set of frames
          (n_channels * n_adiframes). If ``ncomp`` is a float in the interval
          (0, 1] then it corresponds to the desired CEVR, and the corresponding
          number of components will be estimated. If ``ncomp`` is a tuple, then
          it corresponds to an interval of PCs in which final residual frames
          are computed. If ``source_xy`` is not None, then the S/Ns (mean value
          in a 1xFWHM circular aperture) of the given (X,Y) coordinates are
          computed.
        * ADI+mSDI  (``cube`` is a 4d array and ``adimsdi="double"``): ``ncomp``
          must be a tuple, where the first value is the number of PCs obtained
          from each multi-spectral frame (if None then this stage will be
          skipped and the spectral channels will be combined without
          subtraction); the second value sets the number of PCs used in the
          second PCA stage, ADI-like using the residuals of the first stage (if
          None then the second PCA stage is skipped and the residuals are
          de-rotated and combined).
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used.
        * ``lapack``: uses the LAPACK linear algebra library through Numpy
          and it is the most conventional way of computing the SVD
          (deterministic result computed on CPU).
        * ``arpack``: uses the ARPACK Fortran libraries accessible through
          Scipy (computation on CPU).
        * ``eigen``: computes the singular vectors through the
          eigendecomposition of the covariance M.M' (computation on CPU).
        * ``randsvd``: uses the randomized_svd algorithm implemented in
          Sklearn (computation on CPU), proposed in [HAL09]_.
        * ``cupy``: uses the Cupy library for GPU computation of the SVD as in
          the LAPACK version. `
        * ``eigencupy``: offers the same method as with the ``eigen`` option
          but on GPU (through Cupy).
        * ``randcupy``: is an adaptation of the randomized_svd algorithm,
          where all the computations are done on a GPU (through Cupy). `
        * ``pytorch``: uses the Pytorch library for GPU computation of the SVD.
        * ``eigenpytorch``: offers the same method as with the ``eigen``
          option but on GPU (through Pytorch).
        * ``randpytorch``: is an adaptation of the randomized_svd algorithm,
          where all the linear algebra computations are done on a GPU
          (through Pytorch).
    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:
        * ``temp-mean``: temporal px-wise mean is subtracted.
        * ``spat-mean``: spatial mean is subtracted.
        * ``temp-standard``: temporal mean centering plus scaling pixel values
          to unit variance. HIGHLY RECOMMENDED FOR ASDI AND RDI CASES!
        * ``spat-standard``: spatial mean centering plus scaling pixel values
          to unit variance.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
    delta_rot : int, optional
        Factor for tuning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFHWM on each side of the considered frame).
    fwhm : float, list or 1d numpy array, optional
        Known size of the FHWM in pixels to be used. Default value is 4.
        Can be a list or 1d numpy array for a 4d input cube with no scale_list.
    adimsdi : {'single', 'double'}, str optional
        Changes the way the 4d cubes (ADI+mSDI) are processed. Basically it
        determines whether a single or double pass PCA is going to be computed.
        * ``single``: the multi-spectral frames are rescaled wrt the largest
          wavelength to align the speckles and all the frames (n_channels *
          n_adiframes) are processed with a single PCA low-rank approximation.
        * ``double``: a first stage is run on the rescaled spectral frames, and
          a second PCA frame is run on the residuals in an ADI fashion.
    crop_ifs: bool, optional
        [adimsdi='single'] If True cube is cropped at the moment of frame
        rescaling in wavelength. This is recommended for large FOVs such as the
        one of SPHERE, but can remove significant amount of information close to
        the edge of small FOVs (e.g. SINFONI).
    imlib : str, optional
        See the documentation of ``vip_hci.preproc.frame_rotate``.
    imlib2 : str, optional
        See the documentation of ``vip_hci.preproc.cube_rescaling_wavelengths``.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets how temporal residual frames should be combined to produce an
        ADI image.
    collapse_ifs : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets how spectral residual frames should be combined to produce an
        mSDI image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI
        residual channels will be collapsed (by default collapses all channels).
    mask_rdi: 2d numpy array, opt
        If provided, this binary mask will be used either in RDI mode or in
        ADI+mSDI (2 steps) mode. The projection coefficients for the principal
        components will be found considering the area covered by the mask
        (useful to avoid self-subtraction in presence of bright disc signal)
    check_memory : bool, optional
        If True, it checks that the input cube is smaller than the available
        system memory.
    batch : None, int or float, optional
        When it is not None, it triggers the incremental PCA (for ADI and
        ADI+mSDI cubes). If an int is given, it corresponds to the number of
        frames in each sequential mini-batch. If a float (0, 1] is given, it
        corresponds to the size of the batch is computed wrt the available
        memory in the system.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2). Defaults to ``nproc=1``.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints intermediate info and timing.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will subtracted before projecting cube onto reference cube.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "border_mode", "mask_val",
        "edge_blend", "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    frame : numpy ndarray
        2D array, median combination of the de-rotated/re-scaled residuals cube.
        Always returned.
    pcs : numpy ndarray
        [full_output=True, source_xy=None] Principal components. Valid for
        ADI cubes 3D or 4D (i.e. ``scale_list=None``). This is also returned
        when ``batch`` is not None (incremental PCA).
    recon_cube, recon : numpy ndarray
        [full_output=True] Reconstructed cube. Valid for ADI cubes 3D or 4D
        (i.e. ``scale_list=None``)
    residuals_cube : numpy ndarray
        [full_output=True] Residuals cube. Valid for ADI cubes 3D or 4D
        (i.e. ``scale_list=None``)
    residuals_cube_ : numpy ndarray
        [full_output=True] Derotated residuals cube. Valid for ADI cubes 3D or
        4D (i.e. ``scale_list=None``)
    residuals_cube_channels : numpy ndarray
        [full_output=True, adimsdi='double'] Residuals for each multi-spectral
        cube. Valid for ADI+mSDI (4D) cubes (when ``scale_list`` is provided)
    residuals_cube_channels_ : numpy ndarray
        [full_output=True, adimsdi='double'] Derotated final residuals. Valid
        for ADI+mSDI (4D) cubes (when ``scale_list`` is provided)
    cube_allfr_residuals : numpy ndarray
        [full_output=True, adimsdi='single']  Residuals cube (of the big cube
        with channels and time processed together). Valid for ADI+mSDI (4D)
        cubes (when ``scale_list`` is provided)
    cube_adi_residuals : numpy ndarray
        [full_output=True, adimsdi='single'] Residuals cube (of the big cube
        with channels and time processed together) after de-scaling the wls.
        Valid for ADI+mSDI (4D) (when ``scale_list`` is provided).
    medians : numpy ndarray
        [full_output=True, source_xy=None] This is also returned when ``batch``
        is not None (incremental PCA).
    final_residuals_cube : numpy ndarray
        [ncomp is tuple] The residual final PCA frames for a grid a PCs.
    '''

    print('Using full frame PCA from vip_hci :')

    #checks : full_output mandatory, converting lists into arrays
    kwargs['full_output'] = True
    if isinstance(kwargs['cube'], list):
        kwargs['cube'] = np.array(kwargs['cube'])
    if isinstance(kwargs['angle_list'], list):
        kwargs['angle_list'] = np.array(kwargs['angle_list'])

    I_list = pca(**kwargs)[4]

    return(I_list)


def SNR(image, mode='map', telescope_diameter=8, image_ref=None, n_patch=10, i=0, iaz=0, i_star=0, i_planet=1, **kwargs):
    '''
    Signal to Noise Ratio,  output depends on the selected mode.

    image : pymcfost image.Image object

    mode : str, optional
        The recombination method in 'map' or 'planet' or 'diff'

        'planet' : estimate the signal in a fwhm around the planet and the noise in a fwhm-wide annulus at the same separation

        'diff' : estimate the signal in a fwhm around the planet on I and the noise at the same location in I_ref
            I_ref : 2d-array
                Intensity map from the same simulation as I but without the planet. Should not be None if diff.

        'patch' : estimate the signal in a fwhm around the planet and the noise as the rms of 2fwhm-wide patches at the same separation. SNR calculated as described in Mawet+2014.
            n_patch : int, optional
                Total number of patches on which the noise is estimated. Min is 2.

        'map' : plot an SNR map following vip_hci,kw  arguments can be put as kwargs
            array : numpy ndarray
                Input frame (2d array). Is the intensity in image.
            fwhm : float
                Size in pixels of the FWHM. Calculated from image.
            approximated : bool, optional
                If True, an approximated S/N map is generated.
            plot : bool, optional
                If True plots the S/N map. True by default.
            known_sources : None, tuple or tuple of tuples, optional
                To take into account existing sources. It should be a tuple of float/int
                or a tuple of tuples (of float/int) with the coordinate(s) of the known
                sources.
            nproc : int or None
                Number of processes for parallel computing.
            array2 : numpy ndarray, optional
                Additional image (e.g. processed image with negative derotation angles)
                enabling to have more noise samples. Should have the
                same dimensions as array. Taken from image_ref.
            use2alone: bool, optional
                Whether to use array2 alone to estimate the noise (might be useful to
                estimate the snr of extended disk features).
            verbose: bool, optional
                Whether to print timing or not.
            **kwargs : dictionary, optional
                Arguments to be passed to ``plot_frames`` to customize the plot (and to
                save it to disk).

    telescope_diameter : float, optional
        Diameter of the telescope primary mirror in m. Default is VLT.

    image_ref : pymcfost Image.image object, optional
        Reference image for the noise to evaluate the noise. Needed for mode='diff' and eventually for mode='map'.

    i : int, optional
        Inclination index of the mcfost image. Default is 0.

    i_az : int, optional
        Azimuth index of the mcfost image. Default is 0.

    i_star : int, optional
        Star index in the mcfost star list. Default is 0.

    i_planet : int, optional
        Planet index in the mcfost star list. Default is 1.

    **kwargs : for hci_vip.snrmap

    '''

    #Intensity
    I = image.image[0, iaz, i, :, :]
    I_ref = None
    if image_ref is not None :
        I_ref = image_ref.image[0, iaz, i, :, :]

    #fwhm
    fwhm_pix = (((image.wl*1e-6)/telescope_diameter) * 180*3600/np.pi) / image.pixelscale

    #planet position
    x_planet = int(-image.star_positions[0][0][0][i_planet]/image.pixelscale)    #in pixels
    y_planet = int(image.star_positions[1][0][0][i_planet]/image.pixelscale)
    r_planet = np.sqrt( (x_planet)**2 + (y_planet)**2 )

    #separation grids
    halfsize = np.asarray(image.image.shape[-2:]) / 2
    posx = np.linspace(-halfsize[0], halfsize[0], image.nx)
    posy = np.linspace(-halfsize[1], halfsize[1], image.ny)
    meshx, meshy = np.meshgrid(posx, posy)
    r_grid = np.sqrt(meshx ** 2 + meshy ** 2)
    r_grid_planet = np.sqrt((meshx-x_planet) ** 2 + (meshy-y_planet) ** 2)

    #masks
    mask_planet = r_grid_planet <= fwhm_pix
    mask_annulus = np.logical_and(r_planet - 0.5*fwhm_pix <= r_grid, r_grid <= r_planet + 0.5*fwhm_pix)
    mask_noise = np.logical_and(mask_annulus, np.logical_not(mask_planet))

    #patches
    incr_angle = 2*np.pi/(n_patch+1)
    x_patch, y_patch = x_planet, y_planet
    masks_patch = []
    for ind in range(n_patch):
        x_patch_old, y_patch_old = x_patch, y_patch
        x_patch = np.cos(incr_angle)*x_patch_old - np.sin(incr_angle)*y_patch_old
        y_patch = np.sin(incr_angle)*x_patch_old + np.cos(incr_angle)*y_patch_old

        r_grid_patch = np.sqrt((meshx-x_patch)**2 + (meshy-y_patch)**2)
        mask_patch = r_grid_patch <= fwhm_pix
        masks_patch.append(mask_patch)

    #creation of the final noise mask
    mask_patch = masks_patch[0]
    for ind, mask in enumerate(masks_patch):
        if ind != 0 :
            mask_patch = np.logical_or(mask_patch, mask)

    #masks converted into binary arrays
    viz_mask_planet = np.zeros((len(mask_planet), len(mask_planet[0])))
    viz_mask_noise = np.zeros((len(mask_noise), len(mask_noise[0])))
    viz_mask_patch = np.zeros((len(mask_patch), len(mask_patch[0])))
    for ind_x in range(len(mask_planet)):
        for ind_y in range(len(mask_planet[0])):
            if mask_planet[ind_x][ind_y] :
                viz_mask_planet[ind_x][ind_y] = 1

            if mask_noise[ind_x][ind_y] :
                viz_mask_noise[ind_x][ind_y] = 1

            if mask_patch[ind_x][ind_y] :
                viz_mask_patch[ind_x][ind_y] = 1

    viz_masks_patch = []
    for ind_patch in range(n_patch):
        viz_masks_patch.append(np.zeros((len(mask_patch), len(mask_patch[0]))))
        for ind_x in range(len(mask_planet)):
            for ind_y in range(len(mask_planet[0])):
                if masks_patch[ind_patch][ind_x][ind_y] :
                    viz_masks_patch[ind_patch][ind_x][ind_y] = 1

    plt.figure()
    plt.imshow(viz_mask_patch)
    plt.show()

    if mode=='planet':

        print('Calculating the SNR of the planet...')
        I_signal = I*viz_mask_planet
        I_noise = I*viz_mask_noise


        I_signal_list, I_noise_list = [], []
        for elt_i in I_signal :
            for elt_j in elt_i :
                #if elt_j!=0:
                I_signal_list.append(elt_j)
        for elt_i in I_noise :
            for elt_j in elt_i:
                #if elt_j!=0:
                I_noise_list.append(elt_j)

        I_signal_list = np.array(I_signal_list)
        I_noise_list = np.array(I_noise_list)
        #print(len(I_signal_list))
        #print(len(I_noise_list))

        #print(np.median(I_signal_list), np.median(I_noise_list))
        
        #S = np.median(I_signal_list) - np.median(I_noise_list)    #median in a fhwm around the planet - median of the annulus
        #N = np.sqrt(np.mean(I_noise_list*I_noise_list)/len(I_noise_list))    #RMS per pix
        #N = np.std(I_noise_list)/len(I_noise_list)
        
        S = np.sum(I_signal_list)    #signal in a resolution element around the planet
        N = np.sqrt(np.mean(I_noise_list**2)) * len(I_signal_list)  #noise = rms of the annulus * nb of pixels inside a resolution element
        
        SNR = S/N

        print(S, N)

        delta_S = 0
        delta_N = 0
        delta_SNR = 0#np.sqrt( (delta_S/N)**2 + (delta_N * S/N**2)**2 )

        print('SNR =', SNR, ' ; delta_SNR = ', delta_SNR)
        return(SNR, delta_SNR)

    if mode=='patch':
        print('Calculating SNR of the planet from patches...')
        F_signal = np.sum(I*viz_mask_planet)
        F_patch = []
        for patch in viz_masks_patch :
            F_patch.append(np.sum(I*patch))

        S = F_signal - np.mean(F_patch)
        N = np.std(F_patch)*np.sqrt(1+1/n_patch)

        SNR = S/N

        print('SNR =', SNR)
        return(SNR)


    if mode=='map':

        print('Using vip_hci snrmap : ')

        snrmap(I, fwhm=fwhm_pix, plot=True, array2=I_ref, **kwargs)
