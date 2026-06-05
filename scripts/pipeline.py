

import csv, warnings
from itertools import groupby
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy import units as u
import ccdproc as ccdp
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

warnings.filterwarnings("ignore")


FILTERS    = ["r", "g"]   # filters to process; remove one to skip it
APERTURE_R = 12            # aperture radius in pixels
SKY_IN     = 18            # sky annulus inner radius
SKY_OUT    = 26            # sky annulus outer radius

STAR_COORDS = {
    # ("night",     "run label")  : sz pixel pos     , comparison star positions
    ("20260316", "SZLYN")        : {"sz": (2188, 1673), "comp": [(1978,1609),(2483,1711),(1667,1847),(1359,1612),(1464,1906)]},
    ("20260316", "SZLYN-2")      : {"sz": None,          "comp": []},
    ("20260414", "SZLYN")        : {"sz": None,          "comp": []},
    ("20260305", "SZ_Lyncis")    : {"sz": None,          "comp": []},
}

# ============================================================
DATA_DIR = Path("data")
RED_DIR  = Path("reduced")
RED_DIR.mkdir(exist_ok=True)


NIGHTS = {
    "20260316": {
        "bias_glob": "bias-*.fit",
        "dark_glob": "Dark-*_60.fit",
        "flat_glob": {"r": "flat-*_r.fit",  "g": "flat-*_g.fit"},
        "runs": [
            {"label": "SZLYN",   "subdir": "szlyncis",
             "science_glob": {"r": "SZLYN-[0-9]*_r60.fit", "g": "SZLYN-[0-9]*_g60.fit"}},
            {"label": "SZLYN-2", "subdir": "szlyncis2",
             "science_glob": {"r": "SZLYN-2-*_r60.fit",    "g": "SZLYN-2-*_g60.fit"}},
        ],
    },
    "20260414": {
        "bias_glob": "dark-*bias.fit",
        "dark_glob": "dark-*dark.fit",
        "flat_glob": {"r": "flat-*r.fit",   "g": "flat-*g.fit"},
        "runs": [
            {"label": "SZLYN", "subdir": "szlyncis",
             "science_glob": {"r": "sz_lyncis-*r.fit", "g": "sz_lyncis-*g.fit"}},
        ],
    },
    "20260305": {
        "bias_glob": "Calibration-*_bias.fit",
        "dark_glob": "Calibration-*_60s.fit",
        "flat_glob": {"r": "flat-*_r.fit",  "g": "flat-*_g.fit"},
        "runs": [
            {"label": "SZ_Lyncis", "subdir": "szlyncis",
             "science_glob": {"r": "SZ_Lyncis-*_r.fit", "g": "SZ_Lyncis-*_g.fit"}},
        ],
    },
}


def inv_median(a):
    return 1 / np.median(a)

RECALIBRATE = False
all_calibrated = []

for night, cfg in NIGHTS.items():
    cal_path = DATA_DIR / night / "calibration"
    red_path = RED_DIR  / night
    red_path.mkdir(exist_ok=True)

    if not RECALIBRATE:
        for run in cfg["runs"]:
            if not run_red.exists():
                print(f" {night}/{run['label']}: no reduced set RECALIBRATE=True")
                continue
            for filt in FILTERS:
                sci_glob = run["science_glob"].get(filt)
                if not sci_glob:
                    continue
                files = sorted(run_red.glob(sci.glob))
                if not files:
                    continue
                print(f"  {night}/{run['label']}/{filt}: {len(files)} reduced frames loaded")
                for f in files: 
                    ccd = CCDData.read(str(f), unit='adu')
                    all_calibrated.append((night, run["label"], filt, f.name, ccd, ccd.header))
        continue
        
    print(f"\n{'='*60}")
    print(f"  Night: {night}")
    print(f"{'='*60}")

    print(f"combining bias")
    bias_files = sorted(cal_path.glob(cfg["bias_glob"]))
    if not bias_files:
        print(f" no bias files skipping {night}")
        continue

    combined_bias = ccdp.combine(
        [str(f) for f in bias_files],
        method='average',
        sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
        mem_limit=350e6, unit='adu',
    )
    combined_bias.meta['combined'] = True
    combined_bias.uncertainty = None  # drop uncertainty to avoid memory blow-up downstream
    combined_bias.write(red_path / 'combined_bias.fit', overwrite=True)
    print(f"  {len(bias_files)} bias frames -> combined_bias.fit")


    print(f"Calibrating and combining dark frames ---")
    dark_files = sorted(cal_path.glob(cfg["dark_glob"]))
    if not dark_files:
        print(f"  WARNING: no dark files — skipping {night}")
        continue

    combined_dark = ccdp.combine(
        [ccdp.subtract_bias(CCDData.read(str(f), unit='adu'), combined_bias)
         for f in dark_files],
        method='average',
        sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
        mem_limit=350e6,
    )
    combined_dark.meta['combined'] = True
    combined_dark.uncertainty = None 
    combined_dark.write(red_path / 'combined_dark.fit', overwrite=True)
    print(f"  {len(dark_files)} dark frames -> combined_dark.fit")

    combined_flats = {}  

    for filt in FILTERS:
        flat_glob = cfg["flat_glob"].get(filt)
        flat_files = sorted(cal_path.glob(flat_glob)) if flat_glob else []

        if not flat_files:
            print(f"\n  No flat files for filter {filt} in {night} — skipping filter")
            continue

        print(f"Combining flat frames (filter: {filt}) ---")
        calibrated_flats = []
        for f in flat_files:
            ccd = CCDData.read(str(f), unit='adu')
            ccd = ccdp.subtract_bias(ccd, combined_bias)
            ccd = ccdp.subtract_dark(ccd, combined_dark,
                                     exposure_time='EXPTIME', exposure_unit=u.second,
                                     scale=True)
            calibrated_flats.append(ccd)

        combined_flat = ccdp.combine(
            calibrated_flats,
            method='average', scale=inv_median,
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
            mem_limit=350e6,
        )
        combined_flat.meta['combined'] = True
        combined_flat.uncertainty = None  
        combined_flat.write(red_path / f'combined_flat_{filt}.fit', overwrite=True)
        combined_flats[filt] = combined_flat
        print(f"  {len(flat_files)} flat frames -> combined_flat_{filt}.fit")

        # Calibrate science frames 
        for run in cfg["runs"]:
            sci_glob = run["science_glob"].get(filt)
            if not sci_glob:
                continue
            sci_files = sorted((DATA_DIR / night / run["subdir"]).glob(sci_glob))
            if not sci_files:
                print(f"\n  No science files: {night}/{run['label']} filter {filt} — skipping")
                continue

            print(f"\n--- § 6.3  Calibrating science images — {night} / {run['label']} / {filt} ---")
            run_red = red_path / run["subdir"]
            run_red.mkdir(exist_ok=True)

            for f in sci_files:
                ccd = CCDData.read(str(f), unit='adu')
                reduced = ccdp.subtract_bias(ccd, combined_bias)
                reduced = ccdp.subtract_dark(reduced, combined_dark,
                                             exposure_time='EXPTIME', exposure_unit=u.second,
                                             scale=True)
                reduced = ccdp.flat_correct(reduced, combined_flat)
                reduced.write(run_red / f.name, overwrite=True)
                all_calibrated.append((night, run["label"], filt, f.name, reduced, ccd.header))
                print(f"  Calibrated: {f.name}")



if not all_calibrated:
    print("no calibrated frames produced — check file paths.")
else:
    print("Generating field maps")

    seen_runs = set()
    for night, run_label, filt, fname, ccd, hdr in all_calibrated:
        key = (night, run_label)
        if key in seen_runs:
            continue
        seen_runs.add(key)

        data   = ccd.data
        median = np.median(data)
        std    = np.std(data)

        finder  = DAOStarFinder(fwhm=8, threshold=10 * std)
        sources = finder(data - median)

        tag = f"{night}_{run_label}".replace(" ", "_")
        out = f"field_stars_{tag}.png"

        _, ax = plt.subplots(figsize=(14, 14))
        vmin, vmax = np.percentile(data, [1, 99])
        ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        if sources is not None:
            ax.scatter(sources['xcentroid'], sources['ycentroid'],
                       s=40, facecolors='none', edgecolors='cyan', linewidths=0.8)
            for s in sources:
                ax.text(s['xcentroid'] + 15, s['ycentroid'],
                        f"({s['xcentroid']:.0f}, {s['ycentroid']:.0f})",
                        color='yellow', fontsize=5)
        ax.set_title(f"{night} / {run_label} — filter {filt}\n"
                     "Find SZ Lyncis and comparison stars, note their (x, y) coordinates")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        n_src = len(sources) if sources else 0
        print(f"  {out}  ({n_src} stars detected)")


def do_photometry(frames, night, run_label, filt, sz_xy, comp_stars):
    all_positions = [sz_xy] + list(comp_stars)
    rows = []

    for fname, ccd, hdr in frames:
        data = ccd.data

        apertures   = CircularAperture(all_positions, r=APERTURE_R)
        annuli      = CircularAnnulus(all_positions, r_in=SKY_IN, r_out=SKY_OUT)
        phot        = aperture_photometry(data, apertures)
        sky         = aperture_photometry(data, annuli)
        sky_per_pix = sky['aperture_sum'] / annuli.area
        net_flux    = phot['aperture_sum'] - sky_per_pix * apertures.area

        inst_mag  = -2.5 * np.log10(np.maximum(net_flux.value, 1.0))
        comp_mean = float(np.mean(inst_mag[1:]))
        diff_mag  = float(inst_mag[0]) - comp_mean

        hjd = hdr.get('JD-HELIO', hdr.get('JD', 0.0))
        rows.append({'filename': fname, 'HJD': round(hjd, 7), 'diff_mag': round(diff_mag, 5)})
        print(f"  {fname}  HJD={hjd:.6f}  diff_mag={diff_mag:+.4f}")

    tag       = f"{night}_{run_label}_{filt}".replace(" ", "_")
    csv_name  = f"lightcurve_{tag}.csv"
    plot_name = f"lightcurve_{tag}.png"

    with open(csv_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'HJD', 'diff_mag'])
        writer.writeheader()
        writer.writerows(rows)

    hjds  = [r['HJD'] for r in rows]
    mags  = [r['diff_mag'] for r in rows]
    hours = [(t - hjds[0]) * 24 for t in hjds]

    _, ax = plt.subplots(figsize=(10, 5))
    ax.invert_yaxis()
    ax.scatter(hours, mags, s=60, color='steelblue', zorder=3)
    ax.plot(hours, mags, color='steelblue', alpha=0.5)
    ax.set_xlabel(f"Hours since HJD {hjds[0]:.4f}")
    ax.set_ylabel("Differential magnitude (SZ Lyn − comparison ensemble)")
    ax.set_title(f"SZ Lyncis — {night} / {run_label} — {filt} filter")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_name, dpi=150)
    plt.close()
    print(f"  Saved: {csv_name}  {plot_name}")


print("\n\n=== Aperture photometry ===")

for (night, run_label, filt), group in groupby(all_calibrated, key=lambda x: (x[0], x[1], x[2])):
    coords = STAR_COORDS.get((night, run_label))
    if not coords or coords["sz"] is None:
        print(f"\n  Skipping {night} / {run_label} / {filt} — fill in STAR_COORDS to enable")
        continue
    frames = [(fname, ccd, hdr) for _, _, _, fname, ccd, hdr in group]
    print(f"\n  --- {night} / {run_label} / {filt} ({len(frames)} frames) ---")
    do_photometry(frames, night, run_label, filt, coords["sz"], coords["comp"])

print("\nDone!")
