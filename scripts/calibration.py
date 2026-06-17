import sys
import warnings
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy import units as u
from astropy.wcs import FITSFixedWarning
import ccdproc as ccdp

# alleen de cosmetische MJD-OBS/datfix-melding onderdrukken (niet alle warnings)
warnings.simplefilter("ignore", FITSFixedWarning)


# waar de data is opgeslagen
DATA_DIR = Path("data")
RED_DIR  = Path("reduced")

FILTERS = ["r", "g"]

# woordenboek voor de settings
Combine_settings = dict(method='average',
                  sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                  sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                  mem_limit=350e6,
                  )

# data voor nachten en ook wel/niet scalen van flats
NIGHTS = {
    # 03-05  zelfde expt voor flats en darks = niet scalen
    "20260305": {
        "scale_flat": False,
        "bias_glob": "Calibration-*_bias.fit",
        "dark_glob": "Calibration-*_60s.fit",
        "flat_glob": {"r": "flat-*_r.fit", "g": "flat-*_g.fit"},
        "runs": [
            {"subdir": "szlyncis",
             "science_glob": {"r": "SZ_Lyncis-*_r.fit", "g": "SZ_Lyncis-*_g.fit"}},
        ],
    },
    # 03-16  50s darks en 5s flats = wel scalen
    "20260316": {
        "scale_flat": True,
        "bias_glob": "bias-*.fit",
        "dark_glob": "Dark-*.fit",
        "flat_glob": {"r": "flat-*_r.fit", "g": "flat-*_g.fit"},
        "runs": [
            {"subdir": "szlyncis",
             "science_glob": {"r": "SZLYN-[0-9]*_r60.fit", "g": "SZLYN-[0-9]*_g60.fit"}},
            {"subdir": "szlyncis2",
             "science_glob": {"r": "SZLYN-2-*_r60.fit", "g": "SZLYN-2-*_g60.fit"}},
        ],
    },
    # 04-14  60s darks en 10s flats = wel scalen
    "20260414": {
        "scale_flat": True,
        "bias_glob": "dark-*bias.fit",
        "dark_glob": "dark-*dark.fit",
        "flat_glob": {"r": "flat-*r.fit", "g": "flat-*g.fit"},
        "runs": [
            {"subdir": "szlyncis",
             "science_glob": {"r": "sz_lyncis-*r.fit", "g": "sz_lyncis-*g.fit"}},
        ],
    },
}

# leest fitsfile als ccd data
def read_ccd(path):
    return CCDData.read(str(path), unit="adu")

# haalt uit elke fitsfile de exptime
def get_exptime(header):
    for key in ("EXPTIME", "EXPOSURE", "EXPOSED"):
        val = header.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def find_nearest_dark_exposure(image, dark_exposure_times, tolerance=None):
    """
    Find the nearest exposure time of a dark frame to the exposure time of the
    image, raising an error if the difference in exposure time is more than
    tolerance.

    Set tolerance to None to skip the tolerance test.
    """
    dark_exposures = np.array(list(dark_exposure_times))
    image_exposure = get_exptime(image.header) or 0.0
    idx = np.argmin(np.abs(dark_exposures - image_exposure))
    closest_dark_exposure = dark_exposures[idx]

    if (tolerance is not None and
            np.abs(image_exposure - closest_dark_exposure) > tolerance):
        raise RuntimeError('Closest dark exposure time is {} for image of exposure '
                           'time {}.'.format(closest_dark_exposure, image_exposure))

    return closest_dark_exposure


#median 
def inv_median(a):
    return 1 / np.median(a)


# een nacht
def calibrate_night(night, cfg):
    cal_path = DATA_DIR / night / "calibration"
    red_path = RED_DIR / night
    red_path.mkdir(parents=True, exist_ok=True)

    # master bias
    bias_files = sorted(cal_path.glob(cfg["bias_glob"]))
    master_bias = ccdp.combine([str(f) for f in bias_files], unit="adu", **Combine_settings)
    master_bias.meta["combined"] = True
    master_bias.uncertainty = None
    master_bias.data = master_bias.data.astype("float32")   # float32 i.p.v. float64 -> halve bestandsgrootte
    master_bias.write(red_path / "master_bias.fit", overwrite=True)

    # master darks
    dark_files = sorted(cal_path.glob(cfg["dark_glob"]))
    dark_paths_by_et = {}
    for f in dark_files:
        et = get_exptime(fits.getheader(str(f)))     # leest header
        if et is None:
            continue
        dark_paths_by_et.setdefault(round(et, 3), []).append(f)

    master_darks = {}
    for et, paths in sorted(dark_paths_by_et.items()):
        # voor goeie scaling eerst bias aftrekken 
        mdark = ccdp.combine(
            [ccdp.subtract_bias(read_ccd(p), master_bias) for p in paths],
            **Combine_settings,
        )
        mdark.meta["combined"] = True
        mdark.meta["EXPTIME"] = et
        mdark.uncertainty = None
        mdark.data = mdark.data.astype("float32")
        mdark.write(red_path / f"master_dark_{et:g}s.fit", overwrite=True)
        master_darks[et] = mdark

    print(f"masterdark gemaakt voor nacht {night}")

    # master flats
    scale_flat = cfg["scale_flat"]
    master_flats = {}
    for filt in FILTERS:
        pattern = cfg["flat_glob"].get(filt)
        flat_files = sorted(cal_path.glob(pattern)) if pattern else []
        if not flat_files:
            continue

        calibrated_flats = []
        for f in flat_files:
            # bias aftrekken
            ccd = ccdp.subtract_bias(read_ccd(f), master_bias)
            # dichtsbijnde dark vinden 
            closest_dark = find_nearest_dark_exposure(ccd, master_darks.keys(), tolerance=None)
            # dark aftrekken, evt gescaled
            ccd = ccdp.subtract_dark(ccd, master_darks[closest_dark],
                                     exposure_time='exptime', exposure_unit=u.second,
                                     scale=scale_flat)
            calibrated_flats.append(ccd)

        # combineren van de flats
        master_flat = ccdp.combine(calibrated_flats, scale=inv_median, **Combine_settings)
        master_flat.meta["combined"] = True
        master_flat.uncertainty = None
        master_flat.data = master_flat.data.astype("float32")
        master_flat.write(red_path / f"master_flat_{filt}.fit", overwrite=True)
        master_flats[filt] = master_flat

    #frames calibreren
    n_done = 0
    for run in cfg["runs"]:
        sci_dir = DATA_DIR / night / run["subdir"]
        out_dir = red_path / run["subdir"]

        for filt in FILTERS:
            if filt not in master_flats:
                continue
            pattern = run["science_glob"].get(filt)
            sci_files = sorted(sci_dir.glob(pattern)) if pattern else []
            if not sci_files:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            for f in sci_files:
                ccd = read_ccd(f)
                # bias aftrekken
                reduced = ccdp.subtract_bias(ccd, master_bias)
                # goeie dark vinden
                closest_dark = find_nearest_dark_exposure(reduced, master_darks.keys(), tolerance=None)
                # dark aftrekken
                reduced = ccdp.subtract_dark(reduced, master_darks[closest_dark],
                                             exposure_time='exptime', exposure_unit=u.second)
                # correcte flats
                reduced = ccdp.flat_correct(reduced, master_flats[filt])
                reduced.data = reduced.data.astype("float32")
                reduced.write(out_dir / f.name, overwrite=True)
                n_done += 1

    return n_done


# main 
def main():
    for night, cfg in NIGHTS.items():
        try:
            calibrate_night(night, cfg)
        except Exception as exc:
            print(f"{night} fout {exc}")


if __name__ == "__main__":
    main()
