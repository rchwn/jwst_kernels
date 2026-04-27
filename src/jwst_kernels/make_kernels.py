#!/usr/bin/env python
"""JWST PSF Matching Kernel Generator
===================================

Generate PSF matching kernels for JWST MIRI and NIRCam.

Features
--------
- Batch generation of Gaussian kernels
- Batch generation of cross kernels
- Single kernel generation
- Single Gaussian kernel generation
- Aniano-processed PSF generation (--just-processed-psf) with four variants:
    circ_filt     (real-space circularize + Fourier high-pass filter; default)
    circ_nofilt   (real-space circularize only; no Fourier filter)
    nocirc_filt   (Fourier filter only; no real-space circularize)
    nocirc_nofilt (neither circularize nor filter; spatial processing only:
                   interp NaNs, resample, centroid, resize, normalize)
  Select with --psf-variants circ_filt,circ_nofilt,nocirc_filt,nocirc_nofilt
- Processed-to-processed kernel generation (--processed-kernel) between two
  variants of the same band (default: nocirc_nofilt -> circ_filt; configure
  with --from-variant / --to-variant). Reads processed PSFs from the output
  directory and regenerates any missing ones.
- Composite-PSF -> Gaussian kernel generation (--from-composite NAME
  --to-gauss FWHM). The composite is auto-built from composites.toml if
  missing.
- Diagnostic kernel plots (--save-plots) for every kernel that gets
  created (any mode): a 2x2 PNG showing the source PSF, target PSF, the
  kernel itself, and radial profiles annotated with Aniano D / W_-
  performance measures.
- Parallel processing with multiprocessing

Requirements
------------
- jwst_kernels ( https://github.com/francbelf/jwst_kernels )
- astropy
- numpy
- matplotlib

Installing jwst_kernels
-----------------------

git clone https://github.com/francbelf/jwst_kernels
python setup.py develop

Then to run in a modern version of numpy you need to edit
"kernel_core" and replace np.alltrue with np.all

Directory Configuration
-----------------------
Every invocation of the CLI needs both an input PSF directory and an output
kernel directory. Supply them either directly via ``--psf-dir`` and
``--kernel-dir``, or via a TOML config file via ``--config`` with contents like:

    # config.toml
    psf_dir    = "/path/to/psfs/"
    kernel_dir = "/path/to/output/kernels/"

All examples below use ``--config config.toml`` for brevity; you can
equivalently pass ``--psf-dir PATH --kernel-dir PATH``. If you use both,
the explicit ``--psf-dir`` / ``--kernel-dir`` flags override the config.

Batch Processing Usage
----------------------
Process predefined sets of matching kernels in parallel. In batch mode
(without ``--just-processed-psf``) this script produces PSF matching
kernels only; it does NOT produce Aniano-processed source PSFs (use
``--just-processed-psf`` for that, see below).

Cross kernels are only generated in the physically meaningful direction:
from shorter wavelength (sharper PSF) to longer wavelength (broader PSF).
Long -> short wavelength cross kernels are never attempted because you
cannot sharpen a PSF by convolution. Concretely:

    - ``miri``   : MIRI_BANDS[i] -> MIRI_BANDS[j] for all j > i
    - ``nircam`` : NIRCAM_BANDS[i] -> NIRCAM_BANDS[j] for all j > i
    - ``cross``  : NIRCAM_BANDS[i] -> MIRI_BANDS[j] for all i, j
                   (all NIRCam bands are shorter than all MIRI bands)

where MIRI_BANDS and NIRCAM_BANDS are the lists of MIRI and NIRCam bands listed near 
the beginning of the script.

Examples:

    # Process all MIRI bands to Gaussian kernels (0.35", 0.9", 4", 7.5", 15")
    # PLUS all MIRI -> MIRI (short -> long) cross kernels
    python -m jwst_kernels.make_kernels miri --config config.toml -j 8

    # Process all NIRCam bands to Gaussian kernels
    # PLUS all NIRCam -> NIRCam (short -> long) cross kernels
    python -m jwst_kernels.make_kernels nircam --config config.toml -j 8

    # Process NIRCam -> MIRI cross kernels only
    python -m jwst_kernels.make_kernels cross --config config.toml -j 8

    # Process everything: miri + nircam + cross (Gaussian + cross kernels).
    # Does NOT generate Aniano-processed PSFs unless --just-processed-psf is
    # also passed.
    python -m jwst_kernels.make_kernels all --config config.toml -j 8 --overwrite

    # Same, but with paths on the CLI instead of a config file
    python -m jwst_kernels.make_kernels all \
        --psf-dir /path/to/psfs --kernel-dir /path/to/kernels -j 8

Single Kernel Usage
-------------------
Generate individual kernels on demand:

    # Cross kernel: NIRCam F200W to MIRI F770W
    python -m jwst_kernels.make_kernels --from F200W --to F770W \
        --config config.toml

    # Gaussian kernel: MIRI F770W to 7.5" Gaussian
    python -m jwst_kernels.make_kernels --from F770W --to-gauss 7.5 \
        --config config.toml

    # With overwrite, using explicit paths
    python -m jwst_kernels.make_kernels --from F444W --to-gauss 15 \
        --psf-dir /path/to/psfs --kernel-dir /path/to/kernels --overwrite

Aniano-processed PSF Usage
--------------------------
Generate Aniano-processed source PSFs (no kernel):

    # Single band, default variant (circ_filt)
    python -m jwst_kernels.make_kernels --from F335M --just-processed-psf \
        --config config.toml
    python -m jwst_kernels.make_kernels --from F770W --just-processed-psf -o \
        --config config.toml

    # Single band, all four variants in one shot
    python -m jwst_kernels.make_kernels --from F335M --just-processed-psf \
        --config config.toml \
        --psf-variants circ_filt,circ_nofilt,nocirc_filt,nocirc_nofilt

    # Batch all MIRI bands (default variant)
    python -m jwst_kernels.make_kernels miri --just-processed-psf \
        --config config.toml -j 8

    # Batch all bands, only the un-circularized Fourier-filtered variant
    python -m jwst_kernels.make_kernels all --just-processed-psf \
        --config config.toml -j 8 --psf-variants nocirc_filt

    # Batch all bands, spatial-only (no circularize, no Fourier filter)
    python -m jwst_kernels.make_kernels all --just-processed-psf \
        --config config.toml -j 8 --psf-variants nocirc_nofilt

Processed-to-processed Kernel Usage
-----------------------------------
Generate a matching kernel between two processed-PSF variants of the same
band. Source and target variants are configurable via --from-variant and
--to-variant (defaults: nocirc_nofilt -> circ_filt). Missing processed
PSFs are regenerated on demand.

    # Single band, default variants (nocirc_nofilt -> circ_filt). This will make one
    # kernel: F335M_aniano_nocirc_nofilt_to_aniano_circ_filt.fits.
    python -m jwst_kernels.make_kernels --from F335M --processed-kernel \
        --config config.toml

    # Single band, custom variants
    python -m jwst_kernels.make_kernels --from F770W --processed-kernel \
        --from-variant nocirc_filt --to-variant circ_filt \
        --config config.toml

    # Batch all MIRI bands in parallel
    python -m jwst_kernels.make_kernels miri --processed-kernel \
        --config config.toml -j 8

    # Batch everything (miri + nircam)
    python -m jwst_kernels.make_kernels all --processed-kernel \
        --config config.toml -j 8

End-to-end Usage (--all-products)
---------------------------------
Run the full pipeline in a single invocation: Aniano-processed PSFs (both
from/to variants), Gaussian + cross-band matching kernels, and same-band
processed-to-processed kernels, all for the selected camera set.

    # MIRI + NIRCam, default variants (nocirc_nofilt -> circ_filt)
    python -m jwst_kernels.make_kernels all --all-products \
        --config config.toml -j 8

    # MIRI only, custom processed-to-processed variants
    python -m jwst_kernels.make_kernels miri --all-products \
        --from-variant nocirc_filt --to-variant circ_filt \
        --config config.toml -j 8

Composite-PSF -> Gaussian Kernels
---------------------------------
Build a matching kernel that takes a saved composite PSF (as defined in
``composites.toml``, e.g. ``alpha`` = F335M convolved with F300M) to a
Gaussian target. The composite is auto-built from its definition if
missing on disk.

    # Single composite -> Gaussian
    python -m jwst_kernels.make_kernels --from-composite alpha --to-gauss 0.9 \
        --config config.toml

    # All composites in composites.toml -> 0.9" Gaussian, in parallel
    python -m jwst_kernels.make_kernels --composite all --to-gauss 0.9 \
        --config config.toml -j 4

Diagnostic Kernel Plots (--save-plots)
--------------------------------------
For every kernel that gets created (any mode), save a 2x2 diagnostic PNG
showing the source PSF, target PSF, kernel image, and radial profiles
annotated with Aniano D / W_- statistics.

    # Single Gaussian kernel + plot
    python -m jwst_kernels.make_kernels --from F335M --to-gauss 0.9 \
        --config config.toml --save-plots

    # Batch composite -> Gaussian + plots in a custom directory
    python -m jwst_kernels.make_kernels --composite all --to-gauss 0.9 \
        --config config.toml -j 4 --save-plots --plot-dir /tmp/kernel_plots

Output
------
Kernels are saved to the configured output directory with naming:
    - Gaussian: {band}_to_gauss_{fwhm}arcsec.fits
    - Cross: {input_band}_to_{target_band}.fits

Notes
-----
- Camera (MIRI/NIRCam) is automatically detected from band name
- Existing kernels are skipped unless --overwrite is specified
- Parallel processing only applies to batch mode
- All bands must be valid JWST filter names
- Cross kernels are only generated short-wavelength -> long-wavelength
  (high-res -> low-res). The reverse direction is intentionally never
  attempted because convolution cannot sharpen a PSF.
- ``--psf-variants`` only has any effect together with
  ``--just-processed-psf``; passing a non-default value without it is an
  error.
- ``all`` (batch mode, no ``--just-processed-psf``) runs MIRI Gaussian +
  MIRI cross + NIRCam Gaussian + NIRCam cross + NIRCam->MIRI cross. It
  does NOT produce Aniano-processed source PSFs.
- ``all --just-processed-psf`` generates Aniano-processed PSFs for every
  MIRI and NIRCam band (no kernels are produced in that mode).

"""

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import tomllib
from astropy.io import ascii, fits
from astropy import table
import argparse
import multiprocessing as mp
from functools import partial

from jwst_kernels.evaluate_kernels import (
    find_safe_kernel,
    plot_evaluate,
    plot_kernel_diagnostic,
)
from jwst_kernels.make_psf import makeGaussian_2D, read_PSF
from jwst_kernels.kernel_core import (
    MakeConvolutionKernel,
    get_pixscale,
    make_jwst_cross_kernel,
    make_jwst_kernel_to_Gauss,
    plot_kernel,
)

__all__ = [
    "make_jwst_cross_kernel",
    "make_jwst_kernel_to_Gauss",
    "plot_kernel",
    "plot_kernel_diagnostic",
    "find_safe_kernel",
    "plot_evaluate",
    "read_PSF",
    "MakeConvolutionKernel",
    "make_aniano_processed_psf",
    "make_processed_to_processed_kernel",
    "PSF_VARIANTS",
    "load_composites_config",
    "load_component_psf",
    "make_composite_psf",
    "save_composite_psf",
    "process_composite_psfs",
    "make_composite_to_Gauss_kernel",
    "process_composite_gauss",
]

# Set the permissions on the output files

#os.system('umask 002')
os.system('umask 000')

# Default list of target Gaussian FWHWM values
target_gauss_fwhm_list = [0.35, 0.9, 4, 7.5, 15]

# Aniano-processed PSF variants.
# Each variant pairs a real-space circularization flag and a Fourier-domain
# high-pass filter flag, and picks a distinct filename suffix so all three
# variants can coexist on disk.
PSF_VARIANTS = {
    "circ_filt":     {"do_circularize": True,  "do_fourier_filter": True,  "suffix": "aniano_circ_filt"},
    "circ_nofilt":   {"do_circularize": True,  "do_fourier_filter": False, "suffix": "aniano_circ_nofilt"},
    "nocirc_filt":   {"do_circularize": False, "do_fourier_filter": True,  "suffix": "aniano_nocirc_filt"},
    "nocirc_nofilt": {"do_circularize": False, "do_fourier_filter": False, "suffix": "aniano_nocirc_nofilt"},
}
DEFAULT_PSF_VARIANT = "circ_filt"

# All JWST bands (ascending wavelength order)
MIRI_BANDS = [
    'F560W', 'F770W', 'F1000W', 'F1065C', 'F1130W', 'F1140C', 'F1280W',
    'F1500W', 'F1550C', 'F1800W', 'F2100W', 'F2300C', 'F2550W', 'FND',
]

NIRCAM_BANDS = [
    'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 'F162M', 'F164N',
    'F182M', 'F187N', 'F200W', 'F210M', 'F212N', 'F250M', 'F277W', 'F300M',
    'F322W2', 'F323N', 'F335M', 'F356W', 'F360M', 'F405N', 'F410M', 'F430M',
    'F444W', 'F460M', 'F466N', 'F470N', 'F480M',
]

def detect_camera(band):
    """
    Auto-detect which camera a band belongs to.
    
    Parameters
    ----------
    band : str
        Filter/band name (e.g., 'F200W', 'F770W')
    
    Returns
    -------
    str
        'MIRI' or 'NIRCam'
    
    Raises
    ------
    ValueError
        If band cannot be identified
    """
    band_upper = band.upper()
    
    if band_upper in MIRI_BANDS:
        return 'MIRI'
    elif band_upper in NIRCAM_BANDS:
        return 'NIRCam'
    else:
        raise ValueError(f"Cannot determine camera for band '{band}'. "
                        f"Please specify a known JWST filter.")


def check_gaussian_kernel_exists(filt, fwhm, psf_dir, outdir, camera):
    """
    Check if a Gaussian kernel already exists for the given filter and FWHM.
    
    Parameters
    ----------
    filt : str
        Filter name (e.g., 'F770W')
    fwhm : float
        Target FWHM in arcseconds
    camera : str
        Camera name ('MIRI' or 'NIRCam')
    
    Returns
    -------
    bool
        True if kernel file exists, False otherwise
    """

    fwhm_alt = str(fwhm)
    fwhm_alt = fwhm_alt.replace('.','p')
    
    # Check for possible kernel naming patterns
    patterns = [
        f"{outdir}*{filt.lower()}*gauss*{fwhm}*.fits",
        f"{outdir}*{filt.lower()}*gauss*{fwhm_alt}*.fits",
        f"{outdir}*{filt.lower()}*{fwhm}arcsec*.fits",
        f"{outdir}{filt.lower()}*{fwhm}*.fits",
        f"{outdir}{camera.lower()}*{filt.lower()}*{fwhm}*.fits",
        f"{outdir}{camera.lower()}*{filt.lower()}*{fwhm_alt}*.fits",
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if len(matches) > 0:
            return True, matches[0]
    
    return False, None

def check_cross_kernel_exists(input_filt, target_filt, psf_dir, outdir):
    """
    Check if a cross kernel already exists.
    
    Parameters
    ----------
    input_filt : str
        Input filter name
    target_filt : str
        Target filter name
    
    Returns
    -------
    bool
        True if kernel file exists, False otherwise
    """
    pattern = f"{outdir}{input_filt.lower()}*{target_filt.lower()}.fits"
    matches = glob.glob(pattern)
    if len(matches) > 0:
        return True, matches[0]
    return False, None


def make_gaussian_kernel_worker(task):
    """
    Worker function for parallel Gaussian kernel generation.

    Parameters
    ----------
    task : tuple
        (camera, filt, fwhm, psf_dir, outdir, overwrite, save_plots, plot_dir)

    Returns
    -------
    dict
        Result dictionary with status and info
    """
    camera, filt, fwhm, psf_dir, outdir, overwrite, save_plots, plot_dir = task

    try:
        exists, kernel_file = check_gaussian_kernel_exists(
            filt, fwhm, psf_dir, outdir, camera)

        if exists and not overwrite:
            return {
                'success': True,
                'camera': camera,
                'filt': filt,
                'fwhm': fwhm,
                'status': 'skipped',
                'message': f"SKIPPED (exists: {os.path.basename(kernel_file)})"
            }

        status = 'overwriting' if exists else 'creating'

        input_filter = {'camera': camera, 'filter': filt}
        target_gaussian = {'fwhm': fwhm}
        kk = make_jwst_kernel_to_Gauss(input_filter,
                                       target_gaussian,
                                       psf_dir=psf_dir,
                                       outdir=outdir,
                                       detector_effects=True,
                                       save_kernel=True,
                                       save_plots=save_plots,
                                       plot_dir=plot_dir)

        return {
            'success': True,
            'camera': camera,
            'filt': filt,
            'fwhm': fwhm,
            'status': status,
            'message': f"{'OVERWRITTEN' if exists else 'CREATED'}"
        }

    except Exception as e:
        return {
            'success': False,
            'camera': camera,
            'filt': filt,
            'fwhm': fwhm,
            'status': 'error',
            'message': f"ERROR: {str(e)}"
        }


def make_cross_kernel_worker(task):
    """
    Worker function for parallel cross kernel generation.

    Parameters
    ----------
    task : tuple
        (input_filt, target_filt, psf_dir, outdir, overwrite, save_plots, plot_dir)

    Returns
    -------
    dict
        Result dictionary with status and info
    """
    (input_filt, target_filt, psf_dir, outdir, overwrite,
     save_plots, plot_dir) = task

    try:
        exists, kernel_file = check_cross_kernel_exists(input_filt, target_filt, psf_dir, outdir)

        if exists and not overwrite:
            return {
                'success': True,
                'input_filt': input_filt,
                'target_filt': target_filt,
                'status': 'skipped',
                'message': f"SKIPPED (exists: {os.path.basename(kernel_file)})"
            }

        status = 'overwriting' if exists else 'creating'

        input_filter = {'camera': detect_camera(input_filt),
                        'filter': input_filt}
        target_filter = {'camera': detect_camera(target_filt),
                         'filter': target_filt}
        kk = make_jwst_cross_kernel(input_filter,
                                    target_filter,
                                    psf_dir=psf_dir,
                                    outdir=outdir,
                                    detector_effects=True,
                                    save_kernel=True,
                                    save_plots=save_plots,
                                    plot_dir=plot_dir)

        return {
            'success': True,
            'input_filt': input_filt,
            'target_filt': target_filt,
            'status': status,
            'message': f"{'OVERWRITTEN' if exists else 'CREATED'}"
        }

    except Exception as e:
        return {
            'success': False,
            'input_filt': input_filt,
            'target_filt': target_filt,
            'status': 'error',
            'message': f"ERROR: {str(e)}"
        }


def make_aniano_psf_worker(task):
    """
    Worker function for parallel Aniano-processed PSF generation.

    Parameters
    ----------
    task : tuple
        (band, psf_dir, outdir, camera, overwrite, variant_key)
        ``variant_key`` must be a key of :data:`PSF_VARIANTS`.

    Returns
    -------
    dict
        Result dictionary with status and info
    """
    band, psf_dir, outdir, camera, overwrite, variant_key = task
    variant = PSF_VARIANTS[variant_key]
    suffix = variant["suffix"]

    try:
        outfile = os.path.join(str(outdir), f"{band}_{suffix}.fits")
        exists = os.path.isfile(outfile)

        if exists and not overwrite:
            return {
                'success': True,
                'band': band,
                'variant': variant_key,
                'status': 'skipped',
                'message': f"SKIPPED (exists: {os.path.basename(outfile)})"
            }

        status = 'overwriting' if exists else 'creating'

        make_aniano_processed_psf(
            band, psf_dir=psf_dir, outdir=outdir, camera=camera,
            overwrite=overwrite, filename_suffix=suffix,
            do_circularize=variant["do_circularize"],
            do_fourier_filter=variant["do_fourier_filter"],
        )

        return {
            'success': True,
            'band': band,
            'variant': variant_key,
            'status': status,
            'message': f"{'OVERWRITTEN' if exists else 'CREATED'}"
        }

    except Exception as e:
        return {
            'success': False,
            'band': band,
            'variant': variant_key,
            'status': 'error',
            'message': f"ERROR: {str(e)}"
        }


def process_miri_gauss(
        n_procs=1, psf_dir=None, outdir=None, overwrite=False,
        save_plots=False, plot_dir=None):
    """Process MIRI bands to Gaussian kernels

    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    save_plots : bool
        If True, also write a diagnostic PNG for each kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    print("\n=== Processing MIRI to Gaussian ===")
    print(f"Using {n_procs} parallel processes")

    tasks = []
    for filt in MIRI_BANDS:
        for fwhm in target_gauss_fwhm_list:
            tasks.append(('MIRI', filt, fwhm,
                          psf_dir, outdir, overwrite,
                          save_plots, plot_dir))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Process in parallel
    if n_procs > 1:
        with mp.Pool(n_procs) as pool:
            results = pool.map(make_gaussian_kernel_worker, tasks)
    else:
        results = [make_gaussian_kernel_worker(task) for task in tasks]
    
    # Summarize results
    created = sum(1 for r in results if r['status'] in ['creating', 'overwriting'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    # Print detailed results
    for result in results:
        filt = result['filt']
        fwhm = result['fwhm']
        msg = result['message']
        print(f"  {filt} @ {fwhm} arcsec: {msg}")
        
    print(f"\nMIRI Gaussian summary: {created} created, {skipped} skipped, {errors} errors")
    
def process_nircam_gauss(
        n_procs=1, psf_dir=None, outdir=None, overwrite=False,
        save_plots=False, plot_dir=None):
    """Process NIRCam bands to Gaussian kernels

    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    save_plots : bool
        If True, also write a diagnostic PNG for each kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    print("\n=== Processing NIRCam bands ===")
    print(f"Using {n_procs} parallel processes")

    tasks = []
    for filt in NIRCAM_BANDS:
        for fwhm in target_gauss_fwhm_list:
            tasks.append(('NIRCam', filt, fwhm, psf_dir, outdir, overwrite,
                          save_plots, plot_dir))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Process in parallel
    if n_procs > 1:
        with mp.Pool(n_procs) as pool:
            results = pool.map(make_gaussian_kernel_worker, tasks)
    else:
        results = [make_gaussian_kernel_worker(task) for task in tasks]
    
    # Summarize results
    created = sum(1 for r in results if r['status'] in ['creating', 'overwriting'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    # Print detailed results
    for result in results:
        filt = result['filt']
        fwhm = result['fwhm']
        msg = result['message']
        print(f"  {filt} @ {fwhm} arcsec: {msg}")
    
    print(f"\nNIRCam Summary: {created} created, {skipped} skipped, {errors} errors")

def process_miri_cross(
        n_procs=1, psf_dir=None, outdir=None, overwrite=False,
        save_plots=False, plot_dir=None):
    """Process MIRI to MIRI cross kernels.

    Only generates kernels from shorter-wavelength MIRI bands to
    longer-wavelength MIRI bands (i.e. MIRI_BANDS[i] -> MIRI_BANDS[j]
    for j > i). The reverse direction is never attempted because
    convolution cannot sharpen a PSF.

    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    save_plots : bool
        If True, also write a diagnostic PNG for each kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    print("\n=== Processing MIRI to MIRI cross kernels ===")
    print(f"Using {n_procs} parallel processes")

    tasks = []
    for ii, from_filt in enumerate(MIRI_BANDS):
        for jj, to_filt in enumerate(MIRI_BANDS[ii+1:]):
            tasks.append((from_filt, to_filt, psf_dir,
                          outdir, overwrite, save_plots, plot_dir))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Process in parallel
    if n_procs > 1:
        with mp.Pool(n_procs) as pool:
            results = pool.map(make_cross_kernel_worker, tasks)
    else:
        results = [make_cross_kernel_worker(task) for task in tasks]
    
    # Summarize results
    created = sum(1 for r in results if r['status'] in ['creating', 'overwriting'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    # Print detailed results
    for result in results:
        input_filt = result['input_filt']
        target_filt = result['target_filt']
        msg = result['message']
        print(f"  {input_filt} -> {target_filt}: {msg}")
    
    print(f"\nMIRI cross kernels Summary: {created} created, {skipped} skipped, {errors} errors")

def process_nircam_cross(
        n_procs=1, psf_dir=None, outdir=None, overwrite=False,
        save_plots=False, plot_dir=None):
    """Process NIRCam to NIRCam cross kernels.

    Only generates kernels from shorter-wavelength NIRCam bands to
    longer-wavelength NIRCam bands (NIRCAM_BANDS[i] -> NIRCAM_BANDS[j]
    for j > i). The reverse direction is never attempted because
    convolution cannot sharpen a PSF.

    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    save_plots : bool
        If True, also write a diagnostic PNG for each kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    print("\n=== Processing NIRCam to NIRCam cross kernels ===")
    print(f"Using {n_procs} parallel processes")

    tasks = []
    for ii, from_filt in enumerate(NIRCAM_BANDS):
        for jj, to_filt in enumerate(NIRCAM_BANDS[ii+1:]):
            tasks.append((from_filt, to_filt, psf_dir,
                          outdir, overwrite, save_plots, plot_dir))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Process in parallel
    if n_procs > 1:
        with mp.Pool(n_procs) as pool:
            results = pool.map(make_cross_kernel_worker, tasks)
    else:
        results = [make_cross_kernel_worker(task) for task in tasks]
    
    # Summarize results
    created = sum(1 for r in results if r['status'] in ['creating', 'overwriting'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    # Print detailed results
    for result in results:
        input_filt = result['input_filt']
        target_filt = result['target_filt']
        msg = result['message']
        print(f"  {input_filt} -> {target_filt}: {msg}")
    
    print(f"\n NIRCam cross kernels Summary: {created} created, {skipped} skipped, {errors} errors")
    
def process_cross_instrument(
        n_procs=1, psf_dir=None, outdir=None, overwrite=False,
        save_plots=False, plot_dir=None):
    """Process NIRCam to MIRI cross kernels.

    Generates kernels from every NIRCam band to every MIRI band. Because
    all NIRCam bands are at shorter wavelengths than all MIRI bands, this
    is always the physically meaningful short -> long (high-res ->
    low-res) direction. MIRI -> NIRCam kernels are never generated.

    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    save_plots : bool
        If True, also write a diagnostic PNG for each kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    print("\n=== Processing NIRCam to MIRI cross kernels ===")
    print(f"Using {n_procs} parallel processes")

    tasks = []
    for ii, from_filt in enumerate(NIRCAM_BANDS):
        for jj, to_filt in enumerate(MIRI_BANDS):
            tasks.append((from_filt, to_filt, psf_dir,
                          outdir, overwrite, save_plots, plot_dir))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Process in parallel
    if n_procs > 1:
        with mp.Pool(n_procs) as pool:
            results = pool.map(make_cross_kernel_worker, tasks)
    else:
        results = [make_cross_kernel_worker(task) for task in tasks]
    
    # Summarize results
    created = sum(1 for r in results if r['status'] in ['creating', 'overwriting'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    # Print detailed results
    for result in results:
        input_filt = result['input_filt']
        target_filt = result['target_filt']
        msg = result['message']
        print(f"  {input_filt} -> {target_filt}: {msg}")
    
    print(f"\nCross kernels Summary: {created} created, {skipped} skipped, {errors} errors")

def process_aniano_psfs(
        bands, n_procs=1, psf_dir=None, outdir=None, overwrite=False,
        variants=None):
    """Process Aniano-processed PSFs for a list of bands and variants.

    Parameters
    ----------
    bands : list[str]
        List of band names to process
    n_procs : int
        Number of parallel processes to use
    psf_dir : str
        Directory where raw PSFs are stored
    outdir : str
        Directory where processed PSFs are saved
    overwrite : bool
        Whether to overwrite existing files
    variants : list[str], optional
        PSF variant keys to generate (see :data:`PSF_VARIANTS`). Defaults to
        ``[DEFAULT_PSF_VARIANT]`` if not provided.
    """
    if variants is None:
        variants = [DEFAULT_PSF_VARIANT]

    unknown = [v for v in variants if v not in PSF_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown PSF variant(s): {unknown}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )

    print("\n=== Processing Aniano-processed PSFs ===")
    print(f"Using {n_procs} parallel processes")
    print(f"Variants: {variants}")

    tasks = []
    for band in bands:
        camera = detect_camera(band)
        for variant_key in variants:
            tasks.append((band, psf_dir, outdir, camera, overwrite, variant_key))

    print(f"Total tasks: {len(tasks)}")

    if n_procs > 1:
        with mp.Pool(n_procs) as pool:
            results = pool.map(make_aniano_psf_worker, tasks)
    else:
        results = [make_aniano_psf_worker(task) for task in tasks]

    created = sum(1 for r in results if r['status'] in ['creating', 'overwriting'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    for result in results:
        band = result['band']
        variant = result.get('variant', '?')
        msg = result['message']
        print(f"  {band} [{variant}]: {msg}")

    print(f"\nAniano PSF summary: {created} created, {skipped} skipped, {errors} errors")


def make_single_cross_kernel(
        from_band, to_band,
        psf_dir=None, outdir=None,
        overwrite=False, save_plots=False, plot_dir=None):
    """
    Generate a single cross kernel from one band to another.

    Parameters
    ----------
    from_band : str
        Input band name
    to_band : str
        Target band name
    psf_dir : str
        Directory where PSFs are stored
    outdir : str
        Directory where kernels are stored
    overwrite : bool
        Whether to overwrite if exists
    save_plots : bool
        If True, also write a diagnostic PNG for the kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    print(f"\n=== Creating cross kernel: {from_band} -> {to_band} ===")

    from_camera = detect_camera(from_band)
    to_camera = detect_camera(to_band)

    print(f"Detected: {from_band} ({from_camera}) -> {to_band} ({to_camera})")

    exists, kernel_file = check_cross_kernel_exists(from_band, to_band, psf_dir, outdir)
    if exists and not overwrite:
        print(f"Kernel already exists: {kernel_file}")
        print("Use --overwrite to regenerate")
        return
    elif exists:
        print(f"Overwriting existing kernel: {kernel_file}")

    input_filter = {'camera': from_camera, 'filter': from_band}
    target_filter = {'camera': to_camera, 'filter': to_band}

    print("Generating kernel...")
    kk = make_jwst_cross_kernel(input_filter,
                                target_filter,
                                psf_dir=psf_dir,
                                outdir=outdir,
                                detector_effects=True,
                                save_kernel=True,
                                save_plots=save_plots,
                                plot_dir=plot_dir)

    print(f"Kernel created successfully")

def make_single_gaussian_kernel(
        from_band, fwhm, psf_dir=None, outdir=None, overwrite=False,
        save_plots=False, plot_dir=None):
    """
    Generate a single Gaussian kernel from a band to a Gaussian PSF.

    Parameters
    ----------
    from_band : str
        Input band name
    fwhm : float
        Target FWHM in arcseconds
    psf_dir : str
        Directory where PSFs are stored
    outdir : str
        Directory where kernels are stored
    overwrite : bool
        Whether to overwrite if exists
    save_plots : bool
        If True, also write a diagnostic PNG for the kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    print(f"\n=== Creating Gaussian kernel: {from_band} -> Gaussian {fwhm}\" ===")

    camera = detect_camera(from_band)

    print(f"Detected: {from_band} ({camera}) -> Gaussian FWHM = {fwhm} arcsec")

    exists, kernel_file = check_gaussian_kernel_exists(from_band, fwhm, psf_dir, outdir, camera)
    if exists and not overwrite:
        print(f"Kernel already exists: {kernel_file}")
        print("Use --overwrite to regenerate")
        return
    elif exists:
        print(f"Overwriting existing kernel: {kernel_file}")

    input_filter = {'camera': camera, 'filter': from_band}
    target_gaussian = {'fwhm': fwhm}

    print("Generating kernel...")
    kk = make_jwst_kernel_to_Gauss(input_filter,
                                   target_gaussian,
                                   psf_dir=psf_dir,
                                   outdir=outdir,
                                   detector_effects=True,
                                   save_kernel=True,
                                   save_plots=save_plots,
                                   plot_dir=plot_dir)

    print(f"Kernel created successfully")

def make_aniano_processed_psf(band, psf_dir, outdir, camera=None,
                              overwrite=False, filename_suffix='aniano_circ_filt',
                              do_circularize=True, do_fourier_filter=True,
                              **kwargs):
    """Generate an Aniano-processed PSF for a given band.

    Uses jwst_kernels.kernel_core.MakeConvolutionKernel.process_source_psf()
    to apply the Aniano 2011 spatial + (optional) Fourier-domain processing
    pipeline to the source PSF, then saves the result. If the raw PSF file is
    missing, read_PSF will auto-generate it via WebbPSF, forwarding any
    **kwargs (e.g. oversample_factor, fov_arcsec).

    Parameters
    ----------
    band : str
        Input band name (e.g. F335M, F770W).
    psf_dir : str
        Directory where raw PSFs are stored.
    outdir : str
        Directory where the processed PSF is saved.
    camera : str, optional
        Camera name (NIRCam or MIRI). Auto-detected from band if not provided.
    overwrite : bool
        Whether to overwrite an existing output file.
    filename_suffix : str
        Suffix for the output filename (default: 'aniano_circ_filt').
    do_circularize : bool
        If True (default), apply the real-space rotate-and-average circularize
        step during spatial processing. If False, skip it.
    do_fourier_filter : bool
        If True (default), apply the Fourier-domain circularize + high-pass
        filter block. If False, return after spatial processing only.
    **kwargs
        Forwarded to read_PSF (and ultimately to save_miri_PSF /
        save_nircam_PSF if auto-generation is triggered).

    Returns
    -------
    list[str]
        Paths to the saved FITS files.
    """
    if camera is None:
        camera = detect_camera(band)

    print(f"\n=== Creating Aniano-processed PSF: {band} "
          f"(do_circularize={do_circularize}, do_fourier_filter={do_fourier_filter}) ===")

    outfile = os.path.join(str(outdir), f"{band}_{filename_suffix}.fits")
    exists = os.path.isfile(outfile)
    if exists and not overwrite:
        print(f"Aniano-processed PSF already exists: {outfile}")
        print("Use --overwrite to regenerate")
        return
    elif exists:
        print(f"Overwriting existing Aniano-processed PSF: {outfile}")

    source_data, source_pix = read_PSF(
        {"camera": camera, "filter": band}, psf_dir=psf_dir, **kwargs
    )

    ck = MakeConvolutionKernel(
        source_psf=source_data,
        source_pixscale=source_pix,
        source_name=band,
        verbose=True,
    )
    ck.process_source_psf(do_circularize=do_circularize,
                          do_fourier_filter=do_fourier_filter)
    print(f"Processed source shape: {ck.source_psf.shape}")
    print(f"Resolved common_pixscale: {ck.common_pixscale}")
    print(f"Resolved grid_size_arcsec: {ck.grid_size_arcsec}")
    if not np.isclose(ck.common_pixscale, source_pix):
        raise ValueError(
            f"common_pixscale ({ck.common_pixscale}) does not match "
            f"source_pixscale ({source_pix})"
        )

    saved = ck.save_processed_psf(str(outdir), which='source', filename_suffix=filename_suffix)
    print(f"Saved Aniano-processed PSF: {saved[0]}")
    return saved


def _ensure_processed_psf(band, variant_key, psf_dir, outdir, camera=None,
                          overwrite=False, **kwargs):
    """Return the path to a processed PSF FITS file, generating it if missing.

    Parameters
    ----------
    band : str
        Band name (e.g. F335M).
    variant_key : str
        Key into :data:`PSF_VARIANTS`.
    psf_dir : str
        Directory where raw PSFs live (passed through to
        :func:`make_aniano_processed_psf` if regeneration is needed).
    outdir : str
        Directory where processed PSFs are stored.
    camera : str, optional
        Camera name. Auto-detected if None.
    overwrite : bool
        If True, always regenerate.
    **kwargs
        Forwarded to :func:`make_aniano_processed_psf` if regeneration runs.

    Returns
    -------
    str
        Absolute/relative path to the processed PSF FITS file on disk.
    """
    if variant_key not in PSF_VARIANTS:
        raise ValueError(
            f"Unknown PSF variant: {variant_key}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )
    variant = PSF_VARIANTS[variant_key]
    suffix = variant["suffix"]
    outfile = os.path.join(str(outdir), f"{band}_{suffix}.fits")

    if overwrite or not os.path.isfile(outfile):
        print(f"Generating missing/overwrite processed PSF: {outfile}")
        make_aniano_processed_psf(
            band, psf_dir=psf_dir, outdir=outdir, camera=camera,
            overwrite=overwrite, filename_suffix=suffix,
            do_circularize=variant["do_circularize"],
            do_fourier_filter=variant["do_fourier_filter"],
            **kwargs,
        )

    if not os.path.isfile(outfile):
        raise FileNotFoundError(
            f"Processed PSF not found after generation attempt: {outfile}"
        )

    return outfile


def make_processed_to_processed_kernel(
        band, from_variant='nocirc_nofilt', to_variant='circ_filt',
        psf_dir=None, outdir=None, overwrite=False, save_kernel=True,
        verbose=False, save_plots=False, plot_dir=None, **kwargs):
    """Generate a matching kernel between two processed PSF variants of the same band.

    Reads (or regenerates on demand) ``{band}_{from_suffix}.fits`` and
    ``{band}_{to_suffix}.fits`` from ``outdir``, validates that they share the
    same shape and pixel scale, and then builds an Aniano-style matching
    kernel that takes the ``from_variant`` PSF to the ``to_variant`` PSF.
    The spatial-processing pipeline is skipped (the inputs are already
    processed); only the Fourier-domain kernel construction runs.

    Parameters
    ----------
    band : str
        Band name (e.g. F335M, F770W).
    from_variant, to_variant : str
        Keys into :data:`PSF_VARIANTS`. Defaults: ``nocirc_nofilt`` -> ``circ_filt``.
    psf_dir : str
        Directory where raw PSFs live (used only if a processed PSF is missing
        and must be regenerated via :func:`make_aniano_processed_psf`).
    outdir : str
        Directory where processed PSFs are stored and where the output kernel
        is written.
    overwrite : bool
        If True, regenerate processed PSFs (and overwrite an existing output
        kernel) instead of reusing on-disk copies.
    save_kernel : bool
        Whether to write the kernel FITS file.
    verbose : bool
        Passed through to :class:`MakeConvolutionKernel`.
    save_plots : bool
        If True, also write a diagnostic PNG via
        :func:`jwst_kernels.evaluate_kernels.plot_kernel_diagnostic`.
    plot_dir : str, optional
        Directory for diagnostic PNGs. Defaults to ``<outdir>/plots`` when
        ``save_plots=True`` and ``plot_dir`` is None.
    **kwargs
        Forwarded to :func:`make_aniano_processed_psf` when regenerating
        missing processed PSFs (e.g. ``oversample_factor``).

    Returns
    -------
    MakeConvolutionKernel
        The kernel object (with ``kernel`` populated).
    """
    if from_variant not in PSF_VARIANTS:
        raise ValueError(
            f"Unknown from_variant: {from_variant}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )
    if to_variant not in PSF_VARIANTS:
        raise ValueError(
            f"Unknown to_variant: {to_variant}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )
    if from_variant == to_variant:
        raise ValueError(
            f"from_variant and to_variant must differ (got both={from_variant})."
        )

    if psf_dir is None or outdir is None:
        raise ValueError("psf_dir and outdir must be provided.")

    from_suffix = PSF_VARIANTS[from_variant]["suffix"]
    to_suffix = PSF_VARIANTS[to_variant]["suffix"]
    camera = detect_camera(band)

    print(f"\n=== Creating processed-to-processed kernel: "
          f"{band} [{from_variant}] -> {band} [{to_variant}] ===")

    src_file = _ensure_processed_psf(
        band, from_variant, psf_dir=psf_dir, outdir=outdir,
        camera=camera, overwrite=overwrite, **kwargs,
    )
    tgt_file = _ensure_processed_psf(
        band, to_variant, psf_dir=psf_dir, outdir=outdir,
        camera=camera, overwrite=overwrite, **kwargs,
    )

    with fits.open(src_file) as hdul:
        src_data = np.array(hdul[0].data, dtype=float)
        src_pix = get_pixscale(hdul[0].header)
    with fits.open(tgt_file) as hdul:
        tgt_data = np.array(hdul[0].data, dtype=float)
        tgt_pix = get_pixscale(hdul[0].header)

    if src_data.shape != tgt_data.shape:
        raise ValueError(
            f"Processed PSF shapes differ: "
            f"{os.path.basename(src_file)}={src_data.shape} vs "
            f"{os.path.basename(tgt_file)}={tgt_data.shape}. "
            f"Cannot build a kernel between incompatible grids."
        )
    if not np.isclose(src_pix, tgt_pix):
        raise ValueError(
            f"Processed PSF pixel scales differ: "
            f"{os.path.basename(src_file)}={src_pix} vs "
            f"{os.path.basename(tgt_file)}={tgt_pix}. "
            f"Cannot build a kernel between incompatible grids."
        )

    common_pixscale = float(src_pix)
    grid_size_arcsec = np.array(src_data.shape, dtype=float) * common_pixscale
    source_name = f"{band}_{from_suffix}"
    target_name = f"{band}_{to_suffix}"
    kk = MakeConvolutionKernel(
        source_psf=src_data,
        source_pixscale=common_pixscale,
        source_name=source_name,
        target_psf=tgt_data,
        target_pixscale=common_pixscale,
        target_name=target_name,
        common_pixscale=common_pixscale,
        grid_size_arcsec=grid_size_arcsec,
        verbose=verbose,
    )
    kk.make_convolution_kernel_from_processed()

    if save_kernel:
        add_keys = {
            "FROMVAR": (from_variant, "source PSF variant"),
            "TOVAR": (to_variant, "target PSF variant"),
            "BAND": (band, "JWST band"),
        }
        kk.write_out_kernel(outdir=str(outdir), add_keys=add_keys,
                            naming_convention='PHANGS', print_name=True)
    else:
        print("Kernel not saved")

    if save_plots:
        if plot_dir is None:
            plot_dir = os.path.join(str(outdir), 'plots')
        plot_kernel_diagnostic(kk, plot_dir=plot_dir)

    return kk


def check_processed_kernel_exists(band, from_variant, to_variant, outdir):
    """Check whether a processed-to-processed kernel already exists on disk.

    Mirrors the naming used by :meth:`MakeConvolutionKernel.write_out_kernel`
    with ``naming_convention='PHANGS'`` (lowercase ``source_name_to_target_name.fits``).
    """
    from_suffix = PSF_VARIANTS[from_variant]["suffix"]
    to_suffix = PSF_VARIANTS[to_variant]["suffix"]
    source_name = f"{band}_{from_suffix}".lower()
    target_name = f"{band}_{to_suffix}".lower().replace('.', 'p')
    kernel_file = os.path.join(str(outdir), f"{source_name}_to_{target_name}.fits")
    if os.path.isfile(kernel_file):
        return True, kernel_file
    return False, None


def make_processed_kernel_worker(task):
    """Worker function for parallel processed-to-processed kernel generation.

    Parameters
    ----------
    task : tuple
        (band, psf_dir, outdir, overwrite, from_variant, to_variant,
         save_plots, plot_dir)

    Returns
    -------
    dict
        Result dictionary with status and info.
    """
    (band, psf_dir, outdir, overwrite, from_variant, to_variant,
     save_plots, plot_dir) = task

    try:
        exists, kernel_file = check_processed_kernel_exists(
            band, from_variant, to_variant, outdir)

        if exists and not overwrite:
            return {
                'success': True,
                'band': band,
                'from_variant': from_variant,
                'to_variant': to_variant,
                'status': 'skipped',
                'message': f"SKIPPED (exists: {os.path.basename(kernel_file)})"
            }

        status = 'overwriting' if exists else 'creating'

        make_processed_to_processed_kernel(
            band, from_variant=from_variant, to_variant=to_variant,
            psf_dir=psf_dir, outdir=outdir, overwrite=overwrite,
            save_kernel=True,
            save_plots=save_plots, plot_dir=plot_dir,
        )

        return {
            'success': True,
            'band': band,
            'from_variant': from_variant,
            'to_variant': to_variant,
            'status': status,
            'message': f"{'OVERWRITTEN' if exists else 'CREATED'}"
        }

    except Exception as e:
        return {
            'success': False,
            'band': band,
            'from_variant': from_variant,
            'to_variant': to_variant,
            'status': 'error',
            'message': f"ERROR: {str(e)}"
        }


def process_processed_kernels(
        bands, n_procs=1, psf_dir=None, outdir=None, overwrite=False,
        from_variant='nocirc_nofilt', to_variant='circ_filt',
        save_plots=False, plot_dir=None):
    """Batch-process processed-to-processed kernels for a list of bands.

    Parameters
    ----------
    bands : list[str]
        Bands to process.
    n_procs : int
        Number of parallel processes.
    psf_dir, outdir : str
        Input (raw PSF) and output directories.
    overwrite : bool
        Whether to overwrite existing processed PSFs and kernels.
    from_variant, to_variant : str
        Keys into :data:`PSF_VARIANTS`.
    save_plots : bool
        If True, also write a diagnostic PNG for each kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    if from_variant not in PSF_VARIANTS:
        raise ValueError(
            f"Unknown from_variant: {from_variant}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )
    if to_variant not in PSF_VARIANTS:
        raise ValueError(
            f"Unknown to_variant: {to_variant}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )
    if from_variant == to_variant:
        raise ValueError(
            f"from_variant and to_variant must differ (got both={from_variant})."
        )

    print("\n=== Processing processed-to-processed kernels ===")
    print(f"Using {n_procs} parallel processes")
    print(f"Variants: {from_variant} -> {to_variant}")

    tasks = [
        (band, psf_dir, outdir, overwrite, from_variant, to_variant,
         save_plots, plot_dir)
        for band in bands
    ]

    print(f"Total tasks: {len(tasks)}")

    if n_procs > 1:
        with mp.Pool(n_procs) as pool:
            results = pool.map(make_processed_kernel_worker, tasks)
    else:
        results = [make_processed_kernel_worker(task) for task in tasks]

    created = sum(1 for r in results if r['status'] in ['creating', 'overwriting'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    for result in results:
        band = result['band']
        msg = result['message']
        print(f"  {band} [{from_variant} -> {to_variant}]: {msg}")

    print(f"\nProcessed-to-processed kernel summary: "
          f"{created} created, {skipped} skipped, {errors} errors")


# =============================================================================
# COMPOSITE PSF FUNCTIONS
# =============================================================================

def load_composites_config(config_path):
    """Load composite PSF definitions from a TOML file.

    Parameters
    ----------
    config_path : str
        Path to the composites.toml file.

    Returns
    -------
    dict
        Dictionary mapping composite names to their definitions.
        Each definition has 'components' (list) and optional 'description' (str).

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the TOML file is malformed or missing required fields.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Composites config not found: {config_path}")

    with open(config_path, "rb") as f:
        composites = tomllib.load(f)

    for name, recipe in composites.items():
        if "components" not in recipe:
            raise ValueError(
                f"Composite '{name}' missing required 'components' field"
            )
        if not isinstance(recipe["components"], list) or len(recipe["components"]) < 2:
            raise ValueError(
                f"Composite '{name}' must have at least 2 components"
            )
        for i, comp in enumerate(recipe["components"]):
            if "name" not in comp and ("band" not in comp or "variant" not in comp):
                raise ValueError(
                    f"Composite '{name}' component {i}: must have either "
                    "'name' (composite ref) or 'band'+'variant' (processed PSF)"
                )

    return composites


def load_component_psf(component, composites, psf_dir, outdir, overwrite=False,
                       _loading_stack=None):
    """Load a single component PSF (processed band or composite reference).

    Parameters
    ----------
    component : dict
        Component specification. Either:
        - {"band": "FXXXM", "variant": "circ_nofilt"} for a processed PSF
        - {"name": "composite_name"} for a reference to another composite
    composites : dict
        Full composites configuration (for resolving references).
    psf_dir : str
        Directory containing raw PSFs (for regenerating processed PSFs).
    outdir : str
        Directory containing processed and composite PSFs.
    overwrite : bool
        If True, regenerate missing dependencies.
    _loading_stack : set, optional
        Internal use: for keeping track of composite names being loaded to prevent errors.

    Returns
    -------
    data : np.ndarray
        The PSF data array.
    pixscale : float
        Pixel scale in arcsec.
    label : str
        Human-readable label for this component.

    Raises
    ------
    ValueError
        If the component spec is invalid or a circular reference is detected.
    FileNotFoundError
        If the required PSF file cannot be found or generated.
    """
    if _loading_stack is None:
        _loading_stack = set()

    if "name" in component:
        composite_name = component["name"]
        if composite_name in _loading_stack:
            raise ValueError(
                f"Circular reference detected: {composite_name} references itself "
                f"(loading stack: {_loading_stack})"
            )
        if composite_name not in composites:
            raise ValueError(
                f"Unknown composite reference: '{composite_name}'. "
                f"Available: {sorted(composites.keys())}"
            )

        composite_file = os.path.join(outdir, f"composite_{composite_name}.fits")
        if not os.path.isfile(composite_file) or overwrite:
            print(f"  Building dependency: composite '{composite_name}'")
            make_composite_psf(
                composite_name, composites=composites,
                psf_dir=psf_dir, outdir=outdir,
                overwrite=overwrite, _loading_stack=_loading_stack,
            )

        if not os.path.isfile(composite_file):
            raise FileNotFoundError(
                f"Composite PSF not found after build attempt: {composite_file}"
            )

        with fits.open(composite_file) as hdul:
            data = np.array(hdul[0].data, dtype=float)
            pixscale = get_pixscale(hdul[0].header)

        return data, pixscale, f"composite_{composite_name}"

    elif "band" in component and "variant" in component:
        band = component["band"]
        variant = component["variant"]

        if variant not in PSF_VARIANTS:
            raise ValueError(
                f"Unknown variant: '{variant}'. "
                f"Allowed: {sorted(PSF_VARIANTS.keys())}"
            )

        camera = detect_camera(band)
        psf_file = _ensure_processed_psf(
            band, variant, psf_dir=psf_dir, outdir=outdir,
            camera=camera, overwrite=overwrite,
        )

        with fits.open(psf_file) as hdul:
            data = np.array(hdul[0].data, dtype=float)
            pixscale = get_pixscale(hdul[0].header)

        suffix = PSF_VARIANTS[variant]["suffix"]
        return data, pixscale, f"{band}_{suffix}"

    else:
        raise ValueError(
            f"Invalid component spec: {component}. "
            "Must have either 'name' (for composite ref) or "
            "'band' + 'variant' (for processed PSF)."
        )


def make_composite_psf(name, composites, psf_dir, outdir, overwrite=False,
                       _loading_stack=None):
    """Build a composite PSF by convolving its component PSFs.

    Parameters
    ----------
    name : str
        Name of the composite PSF (must be a key in composites).
    composites : dict
        Composite PSF definitions loaded from composites.toml.
    psf_dir : str
        Directory containing raw PSFs.
    outdir : str
        Directory for processed PSFs and output composite PSF.
    overwrite : bool
        If True, regenerate even if output exists.
    _loading_stack : set, optional
        Internal use: for keeping track of composite names being loaded to prevent errors.

    Returns
    -------
    str
        Path to the saved composite PSF FITS file.

    Raises
    ------
    ValueError
        If the composite name is unknown or components are incompatible.
    """
    from scipy.signal import fftconvolve

    if name not in composites:
        raise ValueError(
            f"Unknown composite PSF: '{name}'. "
            f"Available: {sorted(composites.keys())}"
        )

    if _loading_stack is None:
        _loading_stack = set()
    _loading_stack = _loading_stack | {name}

    recipe = composites[name]
    components = recipe["components"]
    description = recipe.get("description", "")

    outfile = os.path.join(outdir, f"composite_{name}.fits")
    if os.path.isfile(outfile) and not overwrite:
        print(f"Composite PSF already exists: {outfile}")
        print("Use --overwrite to regenerate")
        return outfile

    print(f"\n=== Building composite PSF: {name} ===")
    if description:
        print(f"Description: {description}")
    print(f"Components: {len(components)}")

    component_data = []
    component_labels = []
    reference_pixscale = None
    reference_shape = None

    for i, comp in enumerate(components):
        print(f"  Loading component {i + 1}/{len(components)}: {comp}")
        data, pixscale, label = load_component_psf(
            comp, composites=composites,
            psf_dir=psf_dir, outdir=outdir, overwrite=overwrite,
            _loading_stack=_loading_stack,
        )

        if reference_pixscale is None:
            reference_pixscale = pixscale
            reference_shape = data.shape
        else:
            if not np.isclose(pixscale, reference_pixscale, rtol=1e-6):
                raise ValueError(
                    f"Pixel scale mismatch: {label} has pixscale={pixscale}, "
                    f"expected {reference_pixscale} (from {component_labels[0]})"
                )
            if data.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch: {label} has shape={data.shape}, "
                    f"expected {reference_shape} (from {component_labels[0]})"
                )

        component_data.append(data)
        component_labels.append(label)

    print("Convolving components...")
    composite = component_data[0].copy()
    for arr in component_data[1:]:
        composite = fftconvolve(composite, arr, mode='same')

    print("Normalizing to sum=1...")
    composite /= np.nansum(composite)

    print(f"Composite shape: {composite.shape}")
    print(f"Composite pixscale: {reference_pixscale} arcsec/pixel")

    saved_path = save_composite_psf(
        data=composite,
        pixscale=reference_pixscale,
        name=name,
        component_labels=component_labels,
        description=description,
        outdir=outdir,
    )

    print(f"Saved: {saved_path}")
    return saved_path


def save_composite_psf(data, pixscale, name, component_labels, description,
                       outdir):
    """Save a composite PSF to a FITS file.

    Parameters
    ----------
    data : np.ndarray
        The composite PSF data array.
    pixscale : float
        Pixel scale in arcsec.
    name : str
        Name of the composite PSF.
    component_labels : list[str]
        Labels of the component PSFs that were convolved.
    description : str
        Human-readable description.
    outdir : str
        Output directory.

    Returns
    -------
    str
        Path to the saved FITS file.
    """
    os.makedirs(outdir, exist_ok=True)

    hdu = fits.PrimaryHDU(data=np.array(data, dtype=np.float32))
    header = hdu.header

    header['PSFNAME'] = (name, 'Composite PSF name')
    header['PSFTYPE'] = ('composite', 'PSF type (composite = convolution of PSFs)')
    header['PIXELSCL'] = (pixscale, 'arcsec/pixel')

    header['CRPIX1'] = (data.shape[1] + 1) / 2
    header['CRPIX2'] = (data.shape[0] + 1) / 2
    header['CRVAL1'] = 0.0
    header['CRVAL2'] = 0.0
    header['CDELT1'] = -pixscale / 3600
    header['CDELT2'] = pixscale / 3600

    header['NCOMP'] = (len(component_labels), 'Number of component PSFs')
    for i, label in enumerate(component_labels, start=1):
        key = f'COMP{i}'
        header[key] = (label[:68], f'Component {i} identifier')

    if description:
        header['DESCRIPT'] = (description[:68], 'Description')

    outfile = os.path.join(outdir, f"composite_{name}.fits")
    hdu.writeto(outfile, overwrite=True)

    return outfile


def process_composite_psfs(names, composites, psf_dir, outdir, overwrite=False):
    """Batch-process composite PSFs.

    Parameters
    ----------
    names : list[str]
        Composite PSF names to build. Use list(composites.keys()) for all.
    composites : dict
        Composite PSF definitions loaded from composites.toml.
    psf_dir : str
        Directory containing raw PSFs.
    outdir : str
        Directory for output.
    overwrite : bool
        Whether to overwrite existing files.
    """
    print(f"\n=== Processing composite PSFs ===")
    print(f"Composites to build: {names}")

    created = 0
    skipped = 0
    errors = 0

    for name in names:
        try:
            outfile = os.path.join(outdir, f"composite_{name}.fits")
            existed = os.path.isfile(outfile)
            make_composite_psf(
                name, composites=composites,
                psf_dir=psf_dir, outdir=outdir,
                overwrite=overwrite,
            )
            if existed and not overwrite:
                skipped += 1
            else:
                created += 1
        except (ValueError, FileNotFoundError) as e:
            print(f"  ERROR building '{name}': {e}")
            errors += 1

    print(f"\nComposite PSF summary: {created} created, {skipped} skipped, {errors} errors")


def make_composite_to_Gauss_kernel(
        composite_name, fwhm,
        composites=None, composites_config=None,
        psf_dir=None, outdir=None,
        overwrite=False, save_kernel=True, verbose=False,
        save_plots=False, plot_dir=None):
    """Generate a matching kernel from a composite PSF to a Gaussian target.

    Reads (or auto-builds) ``composite_{composite_name}.fits`` from
    ``outdir``, constructs a centred Gaussian on the same grid (same shape
    and pixel scale) with the requested FWHM, then runs the Aniano
    Fourier-domain kernel pipeline (:meth:`MakeConvolutionKernel.make_convolution_kernel_from_processed`).
    The composite is already centred and normalised on a common pixel grid,
    and the Gaussian is built to match, so no additional spatial processing
    is required.

    Parameters
    ----------
    composite_name : str
        Composite PSF name (must be a key in ``composites``).
    fwhm : float
        Target Gaussian FWHM in arcsec. Must be larger than the composite's
        own FWHM (kernel construction cannot sharpen).
    composites : dict, optional
        Composite definitions as returned by :func:`load_composites_config`.
        If None, ``composites_config`` is loaded.
    composites_config : str, optional
        Path to ``composites.toml``. Defaults to ``<outdir>/composites.toml``
        if both ``composites`` and ``composites_config`` are None.
    psf_dir : str
        Directory containing raw PSFs (only used if a component processed PSF
        or the composite itself needs to be regenerated).
    outdir : str
        Directory where composite PSFs live and the output kernel is written.
    overwrite : bool
        If True, regenerate the composite (and overwrite an existing kernel).
    save_kernel : bool
        Whether to write the kernel FITS file.
    verbose : bool
        Forwarded to :class:`MakeConvolutionKernel`.
    save_plots : bool
        If True, also write a diagnostic PNG via
        :func:`jwst_kernels.evaluate_kernels.plot_kernel_diagnostic`.
    plot_dir : str, optional
        Directory for diagnostic PNGs. Defaults to ``<outdir>/plots`` when
        ``save_plots=True`` and ``plot_dir`` is None.

    Returns
    -------
    MakeConvolutionKernel
        Kernel object with ``.kernel`` populated.
    """
    if psf_dir is None or outdir is None:
        raise ValueError("psf_dir and outdir must be provided.")
    if not (fwhm > 0):
        raise ValueError(f"fwhm must be positive, got {fwhm}")

    if composites is None:
        if composites_config is None:
            composites_config = os.path.join(str(outdir), "composites.toml")
        composites = load_composites_config(composites_config)

    if composite_name not in composites:
        raise ValueError(
            f"Unknown composite: '{composite_name}'. "
            f"Available: {sorted(composites.keys())}"
        )

    print(f"\n=== Creating composite-to-Gaussian kernel: "
          f"composite_{composite_name} -> Gaussian {fwhm}\" FWHM ===")

    composite_file = os.path.join(str(outdir), f"composite_{composite_name}.fits")
    if not os.path.isfile(composite_file) or overwrite:
        print(f"Building composite PSF '{composite_name}'")
        make_composite_psf(
            composite_name, composites=composites,
            psf_dir=psf_dir, outdir=outdir, overwrite=overwrite,
        )

    if not os.path.isfile(composite_file):
        raise FileNotFoundError(
            f"Composite PSF not found after build attempt: {composite_file}"
        )

    with fits.open(composite_file) as hdul:
        composite_data = np.array(hdul[0].data, dtype=float)
        composite_pix = get_pixscale(hdul[0].header)

    composite_data = composite_data / np.nansum(composite_data)

    sz_y, sz_x = composite_data.shape
    yy, xx = np.meshgrid(np.arange(sz_x) - (sz_x - 1) / 2,
                         np.arange(sz_y) - (sz_y - 1) / 2)
    sigma_pix = fwhm / 2.355 / composite_pix
    target_psf = makeGaussian_2D((xx, yy), (0, 0), (sigma_pix, sigma_pix))
    target_psf = target_psf / np.nansum(target_psf)

    target_name = 'gauss{:.2f}'.format(fwhm)
    source_name = f"composite_{composite_name}"
    grid_size_arcsec = np.array(composite_data.shape, dtype=float) * composite_pix

    kk = MakeConvolutionKernel(
        source_psf=composite_data,
        source_pixscale=composite_pix,
        source_name=source_name,
        target_psf=target_psf,
        target_pixscale=composite_pix,
        target_fwhm=fwhm,
        target_name=target_name,
        common_pixscale=composite_pix,
        grid_size_arcsec=grid_size_arcsec,
        verbose=verbose,
    )
    kk.make_convolution_kernel_from_processed()

    if save_kernel:
        add_keys = {
            "COMPNAME": (composite_name, "Composite PSF name"),
            "TGTFWHM": (fwhm, "Target Gaussian FWHM (arcsec)"),
        }
        kk.write_out_kernel(outdir=str(outdir), add_keys=add_keys,
                            naming_convention='PHANGS', print_name=True)
    else:
        print("Kernel not saved")

    if save_plots:
        if plot_dir is None:
            plot_dir = os.path.join(str(outdir), 'plots')
        plot_kernel_diagnostic(kk, plot_dir=plot_dir)

    return kk


def check_composite_gauss_kernel_exists(composite_name, fwhm, outdir):
    """Check whether a composite -> Gaussian kernel already exists on disk.

    Mirrors the PHANGS naming convention used by
    :meth:`MakeConvolutionKernel.write_out_kernel` (lowercase
    ``source_name_to_target_name.fits``, with ``"."`` -> ``"p"``).
    """
    target_label = 'gauss{:.2f}'.format(fwhm).replace('.', 'p').lower()
    source_label = f"composite_{composite_name}".lower()
    kernel_file = os.path.join(str(outdir),
                               f"{source_label}_to_{target_label}.fits")
    if os.path.isfile(kernel_file):
        return True, kernel_file
    return False, None


def make_composite_gauss_worker(task):
    """Worker function for parallel composite-to-Gaussian kernel generation.

    Parameters
    ----------
    task : tuple
        (composite_name, fwhm, psf_dir, outdir, overwrite,
         composites_config_path, save_plots, plot_dir)

    Returns
    -------
    dict
        Result dictionary with status and info.
    """
    (composite_name, fwhm, psf_dir, outdir, overwrite,
     composites_config_path, save_plots, plot_dir) = task

    try:
        exists, kernel_file = check_composite_gauss_kernel_exists(
            composite_name, fwhm, outdir)

        if exists and not overwrite:
            return {
                'success': True,
                'composite': composite_name,
                'fwhm': fwhm,
                'status': 'skipped',
                'message': f"SKIPPED (exists: {os.path.basename(kernel_file)})"
            }

        status = 'overwriting' if exists else 'creating'

        composites = load_composites_config(composites_config_path)

        make_composite_to_Gauss_kernel(
            composite_name, fwhm,
            composites=composites,
            psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite, save_kernel=True,
            save_plots=save_plots, plot_dir=plot_dir,
        )

        return {
            'success': True,
            'composite': composite_name,
            'fwhm': fwhm,
            'status': status,
            'message': f"{'OVERWRITTEN' if exists else 'CREATED'}"
        }

    except Exception as e:
        return {
            'success': False,
            'composite': composite_name,
            'fwhm': fwhm,
            'status': 'error',
            'message': f"ERROR: {str(e)}"
        }


def process_composite_gauss(
        composite_names, fwhms, n_procs=1,
        psf_dir=None, outdir=None, overwrite=False,
        composites_config=None,
        save_plots=False, plot_dir=None):
    """Batch-process composite -> Gaussian matching kernels.

    Parameters
    ----------
    composite_names : list[str]
        Composite PSF names to use as kernel sources. Each must be a key in
        the composites config.
    fwhms : list[float]
        Target Gaussian FWHMs in arcsec. Every (composite, fwhm) combination
        becomes one kernel.
    n_procs : int
        Number of parallel processes.
    psf_dir : str
        Directory containing raw PSFs (used only if a composite or its
        component PSFs need to be regenerated).
    outdir : str
        Directory containing composite PSFs and where output kernels are
        written.
    overwrite : bool
        If True, regenerate kernels even if they already exist on disk.
    composites_config : str, optional
        Path to ``composites.toml``. Required so workers can reload it
        independently in each subprocess. Defaults to
        ``<outdir>/composites.toml``.
    save_plots : bool
        If True, also write a diagnostic PNG for each kernel.
    plot_dir : str, optional
        Directory for diagnostic PNGs.
    """
    if not composite_names:
        print("No composites to process.")
        return
    if not fwhms:
        raise ValueError("fwhms must contain at least one value")

    if composites_config is None:
        composites_config = os.path.join(str(outdir), "composites.toml")

    print("\n=== Processing composite -> Gaussian kernels ===")
    print(f"Composites: {composite_names}")
    print(f"Target FWHMs: {fwhms}")
    print(f"Using {n_procs} parallel processes")

    tasks = []
    for name in composite_names:
        for fwhm in fwhms:
            tasks.append((name, fwhm, psf_dir, outdir, overwrite,
                          composites_config, save_plots, plot_dir))

    print(f"Total tasks: {len(tasks)}")

    if n_procs > 1:
        with mp.Pool(n_procs) as pool:
            results = pool.map(make_composite_gauss_worker, tasks)
    else:
        results = [make_composite_gauss_worker(task) for task in tasks]

    created = sum(1 for r in results if r['status'] in ['creating', 'overwriting'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    for result in results:
        name = result['composite']
        fwhm = result['fwhm']
        msg = result['message']
        print(f"  composite_{name} @ {fwhm} arcsec: {msg}")

    print(f"\nComposite -> Gaussian kernel summary: "
          f"{created} created, {skipped} skipped, {errors} errors")


def main():

    parser = argparse.ArgumentParser(
        description='Generate JWST PSF matching kernels for MIRI and/or NIRCam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Configuration:
  Provide directories via CLI args or config file (CLI args override config):
  
  General usage:
        %(prog)s <camera> --config <config_file>
  or equivalently:
        %(prog)s <camera> --psf-dir <psf_dir> --kernel-dir <kernel_dir>

  More examples:

  %(prog)s miri --psf-dir /path/to/psfs --kernel-dir /path/to/kernels
  %(prog)s miri --config my_config.toml
  %(prog)s miri --config my_config.toml --kernel-dir /override/path  # CLI overrides config

Batch Processing Examples:
  Cross kernels are only generated short-wavelength -> long-wavelength
  (the reverse direction is never attempted).

  %(prog)s miri --config config.toml              # MIRI Gaussian + MIRI->MIRI cross kernels
  %(prog)s nircam --psf-dir ./psfs --kernel-dir ./out  # NIRCam Gaussian + NIRCam->NIRCam cross kernels
  %(prog)s miri nircam --config config.toml       # Process both MIRI and NIRCam (no NIRCam->MIRI)
  %(prog)s cross --config config.toml             # NIRCam->MIRI cross kernels only
  %(prog)s all --config config.toml               # Everything: miri + nircam + cross (matching kernels only;
                                                   #   NOT Aniano-processed PSFs unless --just-processed-psf)
  %(prog)s miri --config config.toml --overwrite  # Overwrite existing kernels
  %(prog)s miri --config config.toml -j 8         # Use 8 parallel processes

Single Kernel Examples:
  %(prog)s --from F200W --to F770W --config config.toml       # Cross kernel (NIRCam F200W to MIRI F770W)
  %(prog)s --from F770W --to-gauss 7.5 --config config.toml   # Gaussian kernel (MIRI F770W to Gaussian 7.5")
  %(prog)s --from F444W --to-gauss 15 --psf-dir ./psfs --kernel-dir ./out -o # Gaussian kernel (NIRCam F444W to Gaussian 15")

Aniano-processed PSF Examples:
  %(prog)s --from F335M --just-processed-psf --config config.toml # Aniano-processed PSF (default: circ_filt)
  %(prog)s miri --just-processed-psf --config config.toml -j 8 # Batch MIRI (default: circ_filt)
  %(prog)s --from F335M --just-processed-psf --config config.toml \\
      --psf-variants circ_filt,circ_nofilt,nocirc_filt,nocirc_nofilt # All four variants in one call
  %(prog)s --from F335M --just-processed-psf --config config.toml \\
      --psf-variants nocirc_nofilt # Spatial processing only (no circularize, no Fourier filter)

Processed-to-processed Kernel Examples:
  %(prog)s --from F335M --processed-kernel --config config.toml  # nocirc_nofilt -> circ_filt (default)
  %(prog)s --from F770W --processed-kernel --config config.toml \\
      --from-variant nocirc_filt --to-variant circ_filt          # Custom variant pair
  %(prog)s miri --processed-kernel --config config.toml -j 8     # Batch MIRI
  %(prog)s all --processed-kernel --config config.toml -j 8      # Batch MIRI + NIRCam

End-to-end (--all-products) Examples:
  %(prog)s all  --all-products --config config.toml -j 8         # Processed PSFs + matching kernels + processed kernels
  %(prog)s miri --all-products --config config.toml -j 8 \\
      --from-variant nocirc_filt --to-variant circ_filt          # MIRI only, custom variant pair

Composite PSF Examples:
  %(prog)s --composite alpha --config config.toml                # Build specific composite
  %(prog)s --composite all --config config.toml                  # Build all composites
  %(prog)s --composite all --config config.toml \\
      --composites-config /path/to/composites.toml               # Custom composites file
  %(prog)s --list-composites --composites-config composites.toml # List available composites

Composite -> Gaussian Kernel Examples:
  %(prog)s --from-composite alpha --to-gauss 0.9 --config config.toml  # Single composite -> Gaussian
  %(prog)s --composite all --to-gauss 0.9 --config config.toml -j 4    # Build all composites + composite->0.9" kernels

Diagnostic Plot Examples (works with any kernel mode):
  %(prog)s --from F335M --to-gauss 0.9 --config config.toml --save-plots
  %(prog)s --composite all --to-gauss 0.9 --config config.toml -j 4 \\
      --save-plots --plot-dir /tmp/kernel_plots
  %(prog)s all --config config.toml -j 8 --save-plots
        """)
    
    parser.add_argument('cameras', nargs='*', default=None,
                        choices=['miri', 'nircam', 'cross', 'all'],
                        help=(
                            'Camera set(s) to process in batch mode '
                            '(default: "all" if no --from specified). '
                            '"miri" = MIRI Gaussian + MIRI->MIRI cross kernels; '
                            '"nircam" = NIRCam Gaussian + NIRCam->NIRCam cross kernels; '
                            '"cross" = NIRCam->MIRI cross kernels; '
                            '"all" = miri + nircam + cross. '
                            'Cross kernels are always short->long wavelength only. '
                            'Without --just-processed-psf this produces matching '
                            'kernels only (no Aniano-processed source PSFs).'
                        ))
    
    parser.add_argument('--from', dest='from_band', type=str,
                        help='Source band for single kernel generation')
    
    parser.add_argument('--to', dest='to_band', type=str,
                        help='Target band for cross kernel (use with --from)')
    
    parser.add_argument('--to-gauss', dest='to_gauss', type=float,
                        help='Target Gaussian FWHM in arcsec (use with --from)')
    
    parser.add_argument('--just-processed-psf', action='store_true',
                        help='Only generate Aniano-processed source PSF (not kernel(s)). '
                             'Use with --from for single band, or with camera args for batch.')

    parser.add_argument('--processed-kernel', dest='processed_kernel',
                        action='store_true',
                        help='Generate a matching kernel between two processed PSF '
                             'variants of the same band (source -> target controlled by '
                             '--from-variant and --to-variant). Reads processed PSFs '
                             'from the kernel dir and regenerates any missing ones. '
                             'Mutually exclusive with --to, --to-gauss, and '
                             '--just-processed-psf.')

    parser.add_argument('--all-products', dest='all_products', action='store_true',
                        help='End-to-end batch run: generate Aniano-processed PSFs '
                             '(from_variant + to_variant), Gaussian + cross-band '
                             'matching kernels, and same-band processed-to-processed '
                             'kernels in one invocation. Batch-only (requires a camera '
                             'selector, not --from). Mutually exclusive with '
                             '--just-processed-psf, --processed-kernel, --to, --to-gauss.')

    parser.add_argument('--from-variant', dest='from_variant', type=str,
                        default='nocirc_nofilt',
                        help=(
                            'Source processed-PSF variant for --processed-kernel. '
                            f'Allowed: {",".join(sorted(PSF_VARIANTS.keys()))}. '
                            'Default: nocirc_nofilt.'
                        ))

    parser.add_argument('--to-variant', dest='to_variant', type=str,
                        default='circ_filt',
                        help=(
                            'Target processed-PSF variant for --processed-kernel. '
                            f'Allowed: {",".join(sorted(PSF_VARIANTS.keys()))}. '
                            'Default: circ_filt.'
                        ))

    parser.add_argument('--psf-variants', dest='psf_variants', type=str,
                        default=DEFAULT_PSF_VARIANT,
                        help=(
                            'Comma-separated Aniano-processed PSF variants to generate. '
                            f'Allowed: {",".join(sorted(PSF_VARIANTS.keys()))}. '
                            f'Default: {DEFAULT_PSF_VARIANT}. '
                            'circ_filt = real-space circularize + Fourier high-pass filter; '
                            'circ_nofilt = real-space circularize only (no Fourier filter); '
                            'nocirc_filt = Fourier filter only (no real-space circularize); '
                            'nocirc_nofilt = neither circularize nor filter (spatial processing only: '
                            'interp NaNs, resample, centroid, resize, normalize). '
                            'Only meaningful together with --just-processed-psf.'
                        ))

    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite existing kernel files')
    
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Number of parallel processes to use (default: 1, batch mode only)')

    parser.add_argument('--config', dest='local_config', type=str, default=None,
                        help='.toml config file to set directories (alternative to --psf-dir/--kernel-dir)')
    
    parser.add_argument('--psf-dir', dest='psf_dir', type=str, default=None,
                        help='Directory containing input PSFs (alternative to --config)')
    
    parser.add_argument('--kernel-dir', dest='kernel_dir', type=str, default=None,
                        help='Output directory for kernels (alternative to --config)')

    parser.add_argument('--composite', dest='composite', type=str, default=None,
                        help='Build composite PSF(s): specific name or "all"')

    parser.add_argument('--composites-config', dest='composites_config', type=str,
                        default=None,
                        help='Path to composites.toml (default: composites.toml in kernel_dir)')

    parser.add_argument('--list-composites', dest='list_composites', action='store_true',
                        help='List available composite PSF definitions and exit')

    parser.add_argument('--from-composite', dest='from_composite', type=str,
                        default=None,
                        help='Source composite PSF name for single kernel '
                             'generation (use with --to-gauss). Auto-builds '
                             'the composite if missing.')

    parser.add_argument('--save-plots', dest='save_plots', action='store_true',
                        help='For every kernel that gets created, save a '
                             'diagnostic PNG (source PSF, target PSF, kernel, '
                             'radial profile, and Aniano D / W_- statistics).')

    parser.add_argument('--plot-dir', dest='plot_dir', type=str, default=None,
                        help='Directory for diagnostic plots (default: '
                             '<kernel_dir>/plots). Requires --save-plots.')

    args = parser.parse_args()
    
    # Set global overwrite flag
    overwrite = args.overwrite

    # Handle --list-composites early: only needs --composites-config, not directories.
    if args.list_composites:
        composites_path = args.composites_config
        if composites_path is None:
            if args.kernel_dir is None and args.local_config is None:
                parser.error(
                    "--list-composites requires either --composites-config "
                    "or --kernel-dir/--config to locate composites.toml"
                )
            tmp_outdir = args.kernel_dir
            if tmp_outdir is None and args.local_config is not None:
                try:
                    with open(args.local_config.strip(), "rb") as f:
                        tmp_outdir = tomllib.load(f).get('kernel_dir')
                except (FileNotFoundError, tomllib.TOMLDecodeError):
                    tmp_outdir = None
            if tmp_outdir is None:
                parser.error("Could not resolve composites.toml location")
            composites_path = os.path.join(tmp_outdir, "composites.toml")
        try:
            composites = load_composites_config(composites_path)
        except FileNotFoundError:
            print(f"Composites config not found: {composites_path}")
            print("Create a composites.toml file or specify --composites-config")
            return 1
        except (tomllib.TOMLDecodeError, ValueError) as e:
            print(f"Error loading composites config: {e}")
            return 1

        print(f"Available composite PSFs (from {composites_path}):\n")
        for name, recipe in composites.items():
            desc = recipe.get("description", "(no description)")
            print(f"  {name}")
            print(f"    Description: {desc}")
            print(f"    Components:")
            for comp in recipe["components"]:
                if "name" in comp:
                    print(f"      - composite: {comp['name']}")
                else:
                    print(f"      - {comp['band']} ({comp['variant']})")
            print()
        return 0

    # Parse & validate --psf-variants (only meaningful under --just-processed-psf,
    # but we parse it unconditionally so errors surface early).
    psf_variants = [v.strip() for v in args.psf_variants.split(',') if v.strip()]
    if not psf_variants:
        parser.error("--psf-variants must contain at least one variant")
    unknown_variants = [v for v in psf_variants if v not in PSF_VARIANTS]
    if unknown_variants:
        parser.error(
            f"Unknown --psf-variants: {unknown_variants}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )
    if (args.psf_variants != DEFAULT_PSF_VARIANT) and not args.just_processed_psf:
        parser.error("--psf-variants is only meaningful with --just-processed-psf")

    # Validate --processed-kernel mutual exclusions and variant choices
    if args.processed_kernel:
        if args.just_processed_psf:
            parser.error("--processed-kernel is mutually exclusive with --just-processed-psf")
        if args.to_band:
            parser.error("--processed-kernel is mutually exclusive with --to")
        if args.to_gauss is not None:
            parser.error("--processed-kernel is mutually exclusive with --to-gauss")
    if args.from_variant not in PSF_VARIANTS:
        parser.error(
            f"Unknown --from-variant: {args.from_variant}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )
    if args.to_variant not in PSF_VARIANTS:
        parser.error(
            f"Unknown --to-variant: {args.to_variant}. "
            f"Allowed: {sorted(PSF_VARIANTS.keys())}"
        )
    if args.processed_kernel and args.from_variant == args.to_variant:
        parser.error(
            f"--from-variant and --to-variant must differ "
            f"(got both={args.from_variant})."
        )
    if (args.from_variant != 'nocirc_nofilt' or args.to_variant != 'circ_filt') \
            and not (args.processed_kernel or args.all_products):
        parser.error(
            "--from-variant/--to-variant are only meaningful with "
            "--processed-kernel or --all-products"
        )

    # Validate --all-products mutual exclusions and batch-only requirement
    if args.all_products:
        if args.just_processed_psf:
            parser.error("--all-products is mutually exclusive with --just-processed-psf")
        if args.processed_kernel:
            parser.error("--all-products is mutually exclusive with --processed-kernel")
        if args.from_band:
            parser.error("--all-products is batch-only; remove --from")
        if args.to_band:
            parser.error("--all-products is mutually exclusive with --to")
        if args.to_gauss is not None:
            parser.error("--all-products is mutually exclusive with --to-gauss")
        if args.from_variant == args.to_variant:
            parser.error(
                f"--from-variant and --to-variant must differ "
                f"(got both={args.from_variant})."
            )

    # Validate --from-composite mutual exclusions and required pairings
    if args.from_composite:
        if args.from_band:
            parser.error("--from-composite is mutually exclusive with --from")
        if args.to_band:
            parser.error("--from-composite is mutually exclusive with --to")
        if args.just_processed_psf:
            parser.error("--from-composite is mutually exclusive with --just-processed-psf")
        if args.processed_kernel:
            parser.error("--from-composite is mutually exclusive with --processed-kernel")
        if args.all_products:
            parser.error("--from-composite is mutually exclusive with --all-products")
        if args.composite:
            parser.error("--from-composite is mutually exclusive with --composite "
                         "(use one or the other)")
        if args.to_gauss is None:
            parser.error("--from-composite requires --to-gauss FWHM")

    # Validate --plot-dir requires --save-plots
    if args.plot_dir is not None and not args.save_plots:
        parser.error("--plot-dir requires --save-plots")

    # Set directories from CLI args or config file
    # CLI args take precedence over config file
    psf_dir = args.psf_dir
    outdir = args.kernel_dir

    # Load from config file if provided (CLI args override config values)
    if args.local_config is not None:
        config_file = args.local_config.strip()
        print(f"Reading config file: {config_file}")
        try:
            with open(config_file, "rb") as this_file:
                config_data = tomllib.load(this_file)
        except FileNotFoundError:
            parser.error(f"Config file not found: {config_file}")
        except tomllib.TOMLDecodeError as this_error:
            parser.error(f"TOML parse error in {config_file}: {this_error}")

        if psf_dir is None:
            psf_dir = config_data.get('psf_dir')
        if outdir is None:
            outdir = config_data.get('kernel_dir')

    # Validate that we have both directories
    if psf_dir is None or outdir is None:
        parser.error(
            "Must provide directories via --psf-dir and --kernel-dir, "
            "or via --config with a .toml file containing psf_dir and kernel_dir keys.\n"
            "Example config.toml:\n"
            '  psf_dir = "/path/to/psfs/"\n'
            '  kernel_dir = "/path/to/output/kernels/"'
        )

    print("Using directories:")
    print("... PSF directory: ", psf_dir)
    print("... Kernel directory: ", outdir)

    # Resolve plot directory once up-front so every code path can use it.
    plot_dir = None
    if args.save_plots:
        plot_dir = args.plot_dir or os.path.join(str(outdir), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        print("... Plot directory: ", plot_dir)

    # Resolve number of parallel processes once up-front (used by composite
    # batches as well as the rest of batch mode).
    n_procs = max(1, min(args.jobs, mp.cpu_count()))
    if args.jobs != n_procs:
        print(f"Note: Adjusted number of processes from {args.jobs} to {n_procs}")

    # Handle --from-composite (single composite -> Gaussian kernel)
    if args.from_composite:
        composites_path = args.composites_config
        if composites_path is None:
            composites_path = os.path.join(outdir, "composites.toml")
        try:
            composites = load_composites_config(composites_path)
        except FileNotFoundError:
            print(f"Composites config not found: {composites_path}")
            print("Create a composites.toml file or specify --composites-config")
            return 1
        except (tomllib.TOMLDecodeError, ValueError) as e:
            print(f"Error loading composites config: {e}")
            return 1

        try:
            make_composite_to_Gauss_kernel(
                args.from_composite, args.to_gauss,
                composites=composites,
                composites_config=composites_path,
                psf_dir=psf_dir, outdir=outdir,
                overwrite=overwrite,
                save_plots=args.save_plots, plot_dir=plot_dir,
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            return 1
        print("\n=== Done ===")
        return 0

    # Handle --composite
    if args.composite:
        composites_path = args.composites_config
        if composites_path is None:
            composites_path = os.path.join(outdir, "composites.toml")
        try:
            composites = load_composites_config(composites_path)
        except FileNotFoundError:
            print(f"Composites config not found: {composites_path}")
            print("Create a composites.toml file or specify --composites-config")
            return 1
        except (tomllib.TOMLDecodeError, ValueError) as e:
            print(f"Error loading composites config: {e}")
            return 1

        print(f"Loaded composites from: {composites_path}")

        if args.composite.lower() == "all":
            names_to_build = list(composites.keys())
        else:
            if args.composite not in composites:
                print(f"Unknown composite: '{args.composite}'")
                print(f"Available: {sorted(composites.keys())}")
                return 1
            names_to_build = [args.composite]

        process_composite_psfs(
            names=names_to_build,
            composites=composites,
            psf_dir=psf_dir,
            outdir=outdir,
            overwrite=overwrite,
        )

        if args.to_gauss is not None:
            process_composite_gauss(
                composite_names=names_to_build,
                fwhms=[args.to_gauss],
                n_procs=n_procs,
                psf_dir=psf_dir,
                outdir=outdir,
                overwrite=overwrite,
                composites_config=composites_path,
                save_plots=args.save_plots,
                plot_dir=plot_dir,
            )

        print("\n=== Done ===")
        return 0

    # Single kernel mode
    if args.from_band:
        if args.just_processed_psf:
            try:
                for variant_key in psf_variants:
                    variant = PSF_VARIANTS[variant_key]
                    make_aniano_processed_psf(
                        args.from_band,
                        psf_dir=psf_dir, outdir=outdir,
                        overwrite=overwrite,
                        filename_suffix=variant["suffix"],
                        do_circularize=variant["do_circularize"],
                        do_fourier_filter=variant["do_fourier_filter"],
                    )
            except ValueError as e:
                print(f"Error: {e}")
                return 1
            print("\n=== Done ===")
            return 0

        if args.processed_kernel:
            try:
                make_processed_to_processed_kernel(
                    args.from_band,
                    from_variant=args.from_variant,
                    to_variant=args.to_variant,
                    psf_dir=psf_dir, outdir=outdir,
                    overwrite=overwrite,
                    save_plots=args.save_plots, plot_dir=plot_dir,
                )
            except (ValueError, FileNotFoundError) as e:
                print(f"Error: {e}")
                return 1
            print("\n=== Done ===")
            return 0

        if not args.to_band and args.to_gauss is None:
            parser.error("--from requires either --to, --to-gauss, --just-processed-psf, "
                         "or --processed-kernel")
        if args.to_band and args.to_gauss is not None:
            parser.error("Cannot use both --to and --to-gauss together")

        try:
            if args.to_band:
                make_single_cross_kernel(args.from_band, args.to_band,
                                         psf_dir=psf_dir, outdir=outdir,
                                         overwrite=overwrite,
                                         save_plots=args.save_plots,
                                         plot_dir=plot_dir)
            else:
                make_single_gaussian_kernel(args.from_band, args.to_gauss,
                                            psf_dir=psf_dir, outdir=outdir,
                                            overwrite=overwrite,
                                            save_plots=args.save_plots,
                                            plot_dir=plot_dir)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        print("\n=== Done ===")
        return 0
    
    # Batch mode
    if args.to_band or args.to_gauss is not None:
        parser.error("--to and --to-gauss require --from to be specified")
    
    # Set default cameras if none specified
    if args.cameras is None or len(args.cameras) == 0:
        args.cameras = ['all']

    if overwrite:
        print("*** OVERWRITE MODE: Existing kernels will be regenerated ***\n")
    else:
        print("Overwrite mode is OFF: Will skip existing kernels.\n")
        
    print(f"Using {n_procs} parallel process(es)\n")
    
    # Determine what instruments to process
    cameras = set(args.cameras)
    if 'all' in cameras:
        cameras = {'miri', 'nircam', 'cross'}
    
    if args.just_processed_psf:
        bands = []
        if 'miri' in cameras:
            bands += MIRI_BANDS
        if 'nircam' in cameras:
            bands += NIRCAM_BANDS
        if 'cross' in cameras and 'miri' not in cameras and 'nircam' not in cameras:
            bands += MIRI_BANDS + NIRCAM_BANDS
        process_aniano_psfs(
            bands=bands, n_procs=n_procs,
            psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            variants=psf_variants)
        print("\n=== Done ===")
        return 0

    if args.processed_kernel:
        bands = []
        if 'miri' in cameras:
            bands += MIRI_BANDS
        if 'nircam' in cameras:
            bands += NIRCAM_BANDS
        if 'cross' in cameras and 'miri' not in cameras and 'nircam' not in cameras:
            bands += MIRI_BANDS + NIRCAM_BANDS
        process_processed_kernels(
            bands=bands, n_procs=n_procs,
            psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            from_variant=args.from_variant,
            to_variant=args.to_variant,
            save_plots=args.save_plots, plot_dir=plot_dir)
        print("\n=== Done ===")
        return 0

    if args.all_products:
        print("\n*** --all-products: running full end-to-end pipeline ***")

        # Part 1: processed PSFs for both variants needed by the processed-kernel step
        bands = []
        if 'miri' in cameras:
            bands += MIRI_BANDS
        if 'nircam' in cameras:
            bands += NIRCAM_BANDS
        if 'cross' in cameras and 'miri' not in cameras and 'nircam' not in cameras:
            bands += MIRI_BANDS + NIRCAM_BANDS

        variants_for_psfs = sorted({args.from_variant, args.to_variant})
        print(f"\n--- Part 1/3: Aniano-processed PSFs ({variants_for_psfs}) ---")
        process_aniano_psfs(
            bands=bands, n_procs=n_procs,
            psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            variants=variants_for_psfs)

        # Part 2: Gaussian + cross matching kernels (same as default batch)
        print("\n--- Part 2/3: Gaussian + cross matching kernels ---")
        if 'miri' in cameras:
            process_miri_gauss(
                n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
                overwrite=overwrite,
                save_plots=args.save_plots, plot_dir=plot_dir)
            process_miri_cross(
                n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
                overwrite=overwrite,
                save_plots=args.save_plots, plot_dir=plot_dir)
        if 'nircam' in cameras:
            process_nircam_gauss(
                n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
                overwrite=overwrite,
                save_plots=args.save_plots, plot_dir=plot_dir)
            process_nircam_cross(
                n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
                overwrite=overwrite,
                save_plots=args.save_plots, plot_dir=plot_dir)
        if 'cross' in cameras:
            process_cross_instrument(
                n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
                overwrite=overwrite,
                save_plots=args.save_plots, plot_dir=plot_dir)

        # Part 3: processed-to-processed kernels
        print(f"\n--- Part 3/3: processed-to-processed kernels "
              f"({args.from_variant} -> {args.to_variant}) ---")
        process_processed_kernels(
            bands=bands, n_procs=n_procs,
            psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            from_variant=args.from_variant,
            to_variant=args.to_variant,
            save_plots=args.save_plots, plot_dir=plot_dir)

        print("\n=== Done ===")
        return 0

    if 'miri' in cameras:
        process_miri_gauss(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            save_plots=args.save_plots, plot_dir=plot_dir)

        process_miri_cross(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            save_plots=args.save_plots, plot_dir=plot_dir)

    if 'nircam' in cameras:
        process_nircam_gauss(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            save_plots=args.save_plots, plot_dir=plot_dir)

        process_nircam_cross(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            save_plots=args.save_plots, plot_dir=plot_dir)

    if 'cross' in cameras:
        process_cross_instrument(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            save_plots=args.save_plots, plot_dir=plot_dir)
    
    print("\n=== Done ===")

if __name__ == '__main__':
    main()