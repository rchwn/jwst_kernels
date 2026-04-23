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
- Aniano-processed PSF generation (--just-processed-psf) with three variants:
    circ_filt   (real-space circularize + Fourier high-pass filter; default)
    circ_nofilt (real-space circularize only; no Fourier filter)
    nocirc_filt (Fourier filter only; no real-space circularize)
  Select with --psf-variants circ_filt,circ_nofilt,nocirc_filt
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
Process predefined sets of kernels in parallel:

    # Process all MIRI bands to Gaussian kernels (4", 7.5", 15")
    python -m jwst_kernels.make_kernels miri --config config.toml -j 8

    # Process all NIRCam bands to Gaussian kernels
    python -m jwst_kernels.make_kernels nircam --config config.toml -j 8

    # Process NIRCam to F770W cross kernels
    python -m jwst_kernels.make_kernels cross --config config.toml -j 8

    # Process everything
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

    # Single band, all three variants in one shot
    python -m jwst_kernels.make_kernels --from F335M --just-processed-psf \
        --config config.toml \
        --psf-variants circ_filt,circ_nofilt,nocirc_filt

    # Batch all MIRI bands (default variant)
    python -m jwst_kernels.make_kernels miri --just-processed-psf \
        --config config.toml -j 8

    # Batch all bands, only the un-circularized Fourier-filtered variant
    python -m jwst_kernels.make_kernels all --just-processed-psf \
        --config config.toml -j 8 --psf-variants nocirc_filt

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

from jwst_kernels.evaluate_kernels import find_safe_kernel, plot_evaluate
from jwst_kernels.make_psf import read_PSF
from jwst_kernels.kernel_core import (
    MakeConvolutionKernel,
    make_jwst_cross_kernel,
    make_jwst_kernel_to_Gauss,
    plot_kernel,
)

__all__ = [
    "make_jwst_cross_kernel",
    "make_jwst_kernel_to_Gauss",
    "plot_kernel",
    "find_safe_kernel",
    "plot_evaluate",
    "read_PSF",
    "MakeConvolutionKernel",
    "make_aniano_processed_psf",
    "PSF_VARIANTS",
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
    "circ_filt":   {"do_circularize": True,  "do_fourier_filter": True,  "suffix": "aniano_circ_filt"},
    "circ_nofilt": {"do_circularize": True,  "do_fourier_filter": False, "suffix": "aniano_circ_nofilt"},
    "nocirc_filt": {"do_circularize": False, "do_fourier_filter": True,  "suffix": "aniano_nocirc_filt"},
}
DEFAULT_PSF_VARIANT = "circ_filt"

# All MIRI bands
miri_bands = [
    'F560W',
    'F770W',
    'F1000W',
    'F1130W',
    'F1280W',
    'F1500W',
    'F1800W',
    'F2100W',
    'F2550W',
]

# NIRCam bands
nircam_bands = [
    'F090W',
    'F150W',
    'F187N',
    'F200W',
    'F300M',
    'F335M',
    'F164N',
    'F212N',
    'F277W',
    'F360M',
    'F444W',
    'F405N',
    'F430M',
]

# All JWST bands
MIRI_BANDS = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 
              'F1800W', 'F2100W', 'F2550W', 'F1065C', 'F1140C', 'F1550C', 
              'F2300C', 'FND']

NIRCAM_BANDS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 
                'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F210M', 
                'F212N', 'F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 
                'F335M', 'F356W', 'F360M', 'F405N', 'F410M', 'F430M', 
                'F444W', 'F460M', 'F466N', 'F470N', 'F480M']

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
        (camera, filt, fwhm, psf_dir, outdir, overwrite)
    
    Returns
    -------
    dict
        Result dictionary with status and info
    """
    camera, filt, fwhm, psf_dir, outdir, overwrite = task

    try:
        # Check if kernel already exists
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
        
        # Generate the kernel
        input_filter = {'camera': camera, 'filter': filt}
        target_gaussian = {'fwhm': fwhm}
        kk = make_jwst_kernel_to_Gauss(input_filter,
                                       target_gaussian,
                                       psf_dir=psf_dir,
                                       outdir=outdir,
                                       detector_effects=True,
                                       save_kernel=True)
        
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
        (input_filt, target_filt, psf_dir, outdir, overwrite)
    
    Returns
    -------
    dict
        Result dictionary with status and info
    """
    input_filt, target_filt, psf_dir, outdir, overwrite = task
    
    try:
        # Check if kernel already exists
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
        
        # Generate the kernel
        input_filter = {'camera': detect_camera(input_filt),
                        'filter': input_filt}
        target_filter = {'camera': detect_camera(target_filt),
                         'filter': target_filt}
        kk = make_jwst_cross_kernel(input_filter,
                                    target_filter,
                                    psf_dir=psf_dir,
                                    outdir=outdir,
                                    detector_effects=True,
                                    save_kernel=True)
        
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
        n_procs=1, psf_dir=None, outdir=None, overwrite=False):
    """Process MIRI bands to Gaussian kernels
    
    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    """
    print("\n=== Processing MIRI to Gaussian ===")
    print(f"Using {n_procs} parallel processes")
    
    # Create tasks for all MIRI kernels
    tasks = []
    for filt in miri_bands:
        for fwhm in target_gauss_fwhm_list:
            tasks.append(('MIRI', filt, fwhm,
                          psf_dir, outdir, overwrite))
    
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
        n_procs=1, psf_dir=None, outdir=None, overwrite=False):
    """Process NIRCam bands to Gaussian kernels
    
    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    """
    print("\n=== Processing NIRCam bands ===")
    print(f"Using {n_procs} parallel processes")
    
    # Create tasks for all NIRCam kernels
    tasks = []
    for filt in nircam_bands:
        for fwhm in target_gauss_fwhm_list:
            tasks.append(('NIRCam', filt, fwhm, psf_dir, outdir, overwrite))
    
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
        n_procs=1, psf_dir=None, outdir=None, overwrite=False):
    """Process MIRI to MIRI cross kernels
    
    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    """
    print("\n=== Processing MIRI to MIRI cross kernels ===")
    print(f"Using {n_procs} parallel processes")

    # Create tasks for all cross kernels
    tasks = []
    for ii, from_filt in enumerate(miri_bands):
        for jj, to_filt in enumerate(miri_bands[ii+1:]):
            tasks.append((from_filt, to_filt, psf_dir,
                          outdir, overwrite))
    
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
        n_procs=1, psf_dir=None, outdir=None, overwrite=False):
    """Process NIRCam to NIRCam cross kernels
    
    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    """
    print("\n=== Processing NIRCam to NIRCam cross kernels ===")
    print(f"Using {n_procs} parallel processes")
        
    # Create tasks for all cross kernels
    tasks = []
    for ii, from_filt in enumerate(nircam_bands):
        for jj, to_filt in enumerate(nircam_bands[ii+1:]):
            tasks.append((from_filt, to_filt, psf_dir,
                          outdir, overwrite))
    
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
        n_procs=1, psf_dir=None, outdir=None, overwrite=False):
    """Process NIRCam to MIRI cross kernels
    
    Parameters
    ----------
    n_procs : int
        Number of parallel processes to use
    """
    print("\n=== Processing NIRCam to MIRI cross kernels ===")
    print(f"Using {n_procs} parallel processes")

    # Create tasks for all cross kernels
    tasks = []
    # Create tasks for all cross kernels
    tasks = []
    for ii, from_filt in enumerate(nircam_bands):
        for jj, to_filt in enumerate(miri_bands):
            tasks.append((from_filt, to_filt, psf_dir,
                          outdir, overwrite))
    
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
        overwrite=False):
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
    """
    print(f"\n=== Creating cross kernel: {from_band} -> {to_band} ===")
    
    # Auto-detect cameras
    from_camera = detect_camera(from_band)
    to_camera = detect_camera(to_band)
    
    print(f"Detected: {from_band} ({from_camera}) -> {to_band} ({to_camera})")
    
    # Check if exists
    exists, kernel_file = check_cross_kernel_exists(from_band, to_band, psf_dir, outdir)
    if exists and not overwrite:
        print(f"Kernel already exists: {kernel_file}")
        print("Use --overwrite to regenerate")
        return
    elif exists:
        print(f"Overwriting existing kernel: {kernel_file}")
    
    # Generate kernel
    input_filter = {'camera': from_camera, 'filter': from_band}
    target_filter = {'camera': to_camera, 'filter': to_band}
    
    print("Generating kernel...")
    kk = make_jwst_cross_kernel(input_filter,
                                target_filter,
                                psf_dir=psf_dir,
                                outdir=outdir,
                                detector_effects=True,
                                save_kernel=True)
    
    print(f"✓ Kernel created successfully")

def make_single_gaussian_kernel(
        from_band, fwhm, psf_dir=None, outdir=None, overwrite=False):
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
    """
    print(f"\n=== Creating Gaussian kernel: {from_band} -> Gaussian {fwhm}\" ===")
    
    # Auto-detect camera
    camera = detect_camera(from_band)
    
    print(f"Detected: {from_band} ({camera}) -> Gaussian FWHM = {fwhm} arcsec")
    
    # Check if exists
    exists, kernel_file = check_gaussian_kernel_exists(from_band, fwhm, psf_dir, outdir, camera)
    if exists and not overwrite:
        print(f"Kernel already exists: {kernel_file}")
        print("Use --overwrite to regenerate")
        return
    elif exists:
        print(f"Overwriting existing kernel: {kernel_file}")
    
    # Generate kernel
    input_filter = {'camera': camera, 'filter': from_band}
    target_gaussian = {'fwhm': fwhm}
    
    print("Generating kernel...")
    kk = make_jwst_kernel_to_Gauss(input_filter,
                                   target_gaussian,
                                   psf_dir=psf_dir,
                                   outdir=outdir,
                                   detector_effects=True,
                                   save_kernel=True)
    
    print(f"✓ Kernel created successfully")

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


def main():

    parser = argparse.ArgumentParser(
        description='Generate JWST PSF matching kernels for MIRI and/or NIRCam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Configuration:
  Provide directories via CLI args or config file (CLI args override config):
  
  %(prog)s miri --psf-dir /path/to/psfs --kernel-dir /path/to/kernels
  %(prog)s miri --config my_config.toml
  %(prog)s miri --config my_config.toml --kernel-dir /override/path  # CLI overrides config

Batch Processing Examples:
  %(prog)s miri --config config.toml              # Process only MIRI bands
  %(prog)s nircam --psf-dir ./psfs --kernel-dir ./out  # NIRCam with direct paths
  %(prog)s miri nircam --config config.toml       # Process both MIRI and NIRCam
  %(prog)s cross --config config.toml             # Process only cross kernels
  %(prog)s all --config config.toml               # Process everything (default)
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
      --psf-variants circ_filt,circ_nofilt,nocirc_filt # All three variants in one call
        """)
    
    parser.add_argument('cameras', nargs='*', default=None,
                        choices=['miri', 'nircam', 'cross', 'all'],
                        help='Camera(s) to process for batch mode (default: all if no --from specified)')
    
    parser.add_argument('--from', dest='from_band', type=str,
                        help='Source band for single kernel generation')
    
    parser.add_argument('--to', dest='to_band', type=str,
                        help='Target band for cross kernel (use with --from)')
    
    parser.add_argument('--to-gauss', dest='to_gauss', type=float,
                        help='Target Gaussian FWHM in arcsec (use with --from)')
    
    parser.add_argument('--just-processed-psf', action='store_true',
                        help='Only generate Aniano-processed source PSF (not kernel(s)). '
                             'Use with --from for single band, or with camera args for batch.')

    parser.add_argument('--psf-variants', dest='psf_variants', type=str,
                        default=DEFAULT_PSF_VARIANT,
                        help=(
                            'Comma-separated Aniano-processed PSF variants to generate. '
                            f'Allowed: {",".join(sorted(PSF_VARIANTS.keys()))}. '
                            f'Default: {DEFAULT_PSF_VARIANT}. '
                            'circ_filt = real-space circularize + Fourier high-pass filter; '
                            'circ_nofilt = real-space circularize only (no Fourier filter); '
                            'nocirc_filt = Fourier filter only (no real-space circularize). '
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
    
    args = parser.parse_args()
    
    # Set global overwrite flag
    overwrite = args.overwrite

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

        if not args.to_band and args.to_gauss is None:
            parser.error("--from requires either --to, --to-gauss, or --just-processed-psf")
        if args.to_band and args.to_gauss is not None:
            parser.error("Cannot use both --to and --to-gauss together")
        
        try:
            if args.to_band:
                make_single_cross_kernel(args.from_band, args.to_band,
                                         psf_dir=psf_dir, outdir=outdir,
                                         overwrite=overwrite)
            else:
                make_single_gaussian_kernel(args.from_band, args.to_gauss,
                                            psf_dir=psf_dir, outdir=outdir,
                                            overwrite=overwrite)
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
    
    # Validate number of jobs
    n_procs = max(1, min(args.jobs, mp.cpu_count()))
    if args.jobs != n_procs:
        print(f"Note: Adjusted number of processes from {args.jobs} to {n_procs}")
    
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
            bands += miri_bands
        if 'nircam' in cameras:
            bands += nircam_bands
        if 'cross' in cameras and 'miri' not in cameras and 'nircam' not in cameras:
            bands += miri_bands + nircam_bands
        process_aniano_psfs(
            bands=bands, n_procs=n_procs,
            psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite,
            variants=psf_variants)
        print("\n=== Done ===")
        return 0

    if 'miri' in cameras:
        process_miri_gauss(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite)
        
        process_miri_cross(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite)
    
    if 'nircam' in cameras:
        process_nircam_gauss(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite)
        
        process_nircam_cross(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite)
   
    if 'cross' in cameras:
        process_cross_instrument(
            n_procs=n_procs, psf_dir=psf_dir, outdir=outdir,
            overwrite=overwrite)
    
    print("\n=== Done ===")

if __name__ == '__main__':
    main()