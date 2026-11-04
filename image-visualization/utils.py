"""Async cutout cache, plus the masking used by the example profiles."""

import errno
import inspect
import os
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.isophote import Ellipse, EllipseSample, Isophote, IsophoteList
from photutils.segmentation import deblend_sources, detect_sources, detect_threshold

# photutils 2.x: nsigma/npixels/nlevels/nclip; 3.x: n_sigma/n_pixels/n_levels/n_clip.
_DETECT_NSIGMA = (
    "n_sigma" if "n_sigma" in inspect.signature(detect_threshold).parameters else "nsigma"
)
_DETECT_NPIXELS = (
    "n_pixels" if "n_pixels" in inspect.signature(detect_sources).parameters else "npixels"
)
_DEBLEND_SIG = inspect.signature(deblend_sources).parameters
_DEBLEND_NPIXELS = "n_pixels" if "n_pixels" in _DEBLEND_SIG else "npixels"
_DEBLEND_NLEVELS = "n_levels" if "n_levels" in _DEBLEND_SIG else "nlevels"
_NCLIP_KW = (
    "n_clip" if "n_clip" in inspect.signature(Ellipse.fit_image).parameters else "nclip"
)


def isophote_clip_kwargs(sclip=3.0, n_clip=2):
    """Keyword args for Ellipse.fit_image / EllipseSample across photutils 2/3."""
    return {"sclip": sclip, _NCLIP_KW: n_clip}


def ensure_cache_dir(cache_dir="cutouts"):
    """Create a writable cache directory.

    Shared JupyterHub volumes sometimes leave an NFS stale handle (errno 116)
    on a previously deleted folder. Fall back to a process-local name then.
    """
    cache_dir = Path(cache_dir)
    try:
        cache_dir.mkdir(exist_ok=True)
        os.listdir(cache_dir)
        return cache_dir
    except OSError as exc:
        if getattr(exc, "errno", None) not in {errno.ESTALE, 116}:
            raise
    alt = cache_dir.with_name(f"{cache_dir.name}_{os.getpid()}")
    alt.mkdir(exist_ok=True)
    warnings.warn(
        f"{cache_dir} has a stale NFS handle; using {alt} instead.",
        RuntimeWarning,
        stacklevel=2,
    )
    return alt


def resolve_cutout_band(result, tasks, bands):
    """Identify the filter of one cutout result.

    Result rows follow request order. Task rows do not, so the band has to
    come from the matching task sequence or from the result id.
    """
    sequence = result.get("sequence")
    task = next((item for item in tasks if item.get("sequence") == sequence), None)
    task_band = task.get("band") if task else None

    text = f"{result.get('result_id', '')} {result.get('download_url', '')}"
    url_band = next(
        (
            band
            for band in sorted(bands, key=len, reverse=True)
            if f"_{band}_" in text or f"_{band}." in text
        ),
        None,
    )

    if task_band in bands:
        if url_band and url_band != task_band:
            warnings.warn(
                f"Result sequence {sequence} says {task_band}, but the URL looks like {url_band}."
            )
        return task_band
    if url_band:
        return url_band
    raise RuntimeError(f"Could not identify the band for result {result.get('result_id')}.")


def cache_band_cutout(session, result, tasks, bands, cache_dir, radius_arcmin):
    """Save one cutout under its filter name, replacing a mismatched cache file."""
    band = resolve_cutout_band(result, tasks, bands)
    stem = f"ic1816_des_dr2_{radius_arcmin:g}arcmin"
    path = Path(cache_dir) / f"{stem}_{band}.fits"
    if path.exists() and fits.getheader(path).get("BAND") != band:
        warnings.warn(f"{path.name} is cached under the wrong filter; downloading again.")
        path.unlink()
    if not path.exists():
        response = session.get(result["download_url"], timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)

    header_band = fits.getheader(path).get("BAND")
    if header_band in bands and header_band != band:
        warnings.warn(f"{path.name} header BAND is {header_band}; using the header.")
        corrected = path.with_name(f"{stem}_{header_band}.fits")
        if corrected.exists():
            corrected.unlink()
        path.replace(corrected)
        band, path = header_band, corrected
    return band, path


def prepare_band_image(fits_path):
    """Read one cutout, mask contaminants, and subtract a local sky level.

    Returns the sky-subtracted masked image, magnitude zero-point, pixel
    scale in arcsec/pixel, and the uncertainty of the mean sky.
    """
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.astype(np.float64)

    if "MAGZERO" in hdr:
        zp = float(hdr["MAGZERO"])
    elif "ZP" in hdr:
        zp = float(hdr["ZP"])
    else:
        warnings.warn(f"MAGZERO/ZP not found in {fits_path.name}; defaulting to 30.0.")
        zp = 30.0

    pix_scale = float(np.mean(proj_plane_pixel_scales(WCS(hdr))) * 3600.0)

    edge_mask = (data == 0.0) | ~np.isfinite(data)

    threshold = detect_threshold(
        data, **{_DETECT_NSIGMA: 3.0}, sigma_clip=SigmaClip(3.0), mask=edge_mask
    )
    segm = detect_sources(
        np.nan_to_num(data), threshold, **{_DETECT_NPIXELS: 15}, mask=edge_mask
    )
    source_mask = np.zeros_like(edge_mask)
    if segm is not None:
        deblend_kw = {
            _DEBLEND_NPIXELS: 15,
            _DEBLEND_NLEVELS: 32,
            "contrast": 0.005,
        }
        if "progress_bar" in _DEBLEND_SIG:
            deblend_kw["progress_bar"] = False
        segm_deblend = deblend_sources(np.nan_to_num(data), segm, **deblend_kw)
        yc, xc = data.shape[0] // 2, data.shape[1] // 2
        central_label = segm_deblend.data[yc, xc]
        if central_label == 0:
            box = segm_deblend.data[yc - 20 : yc + 20, xc - 20 : xc + 20]
            labels = box[box > 0]
            central_label = np.bincount(labels).argmax() if labels.size else 0
        source_mask = (segm_deblend.data > 0) & (segm_deblend.data != central_label)

    full_mask = edge_mask | source_mask

    yy, xx = np.indices(data.shape)
    yc, xc = data.shape[0] / 2.0, data.shape[1] / 2.0
    r_grid = np.hypot(xx - xc, yy - yc)
    sky_zone = (r_grid > 0.85 * min(xc, yc)) & ~full_mask
    if np.sum(sky_zone) < 100:
        sky_zone = ~full_mask

    _, median_bg, std_bg = sigma_clipped_stats(data[sky_zone], sigma=3.0, maxiters=5)
    sky_err = std_bg / np.sqrt(max(np.sum(sky_zone), 1))

    masked_sub = np.ma.masked_array(data - median_bg, mask=full_mask)
    return masked_sub, zp, pix_scale, sky_err


def extract_forced_isophotes(masked_img, ref_isolist):
    """Sample a band on the fixed ellipse geometries of the reference fit."""
    forced = []
    for iso in ref_isolist:
        if iso.sma == 0:
            continue
        sample = EllipseSample(
            masked_img, iso.sma, geometry=iso.sample.geometry, **isophote_clip_kwargs()
        )
        sample.update(fixed_parameters=np.array([True] * 4))
        forced.append(Isophote(sample, 0, True, 0))
    return IsophoteList(forced)
