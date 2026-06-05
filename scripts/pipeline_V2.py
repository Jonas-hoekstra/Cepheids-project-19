import csv
import warnings
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
from photutils.centroids import centroid_com

warnings.filterwarnings("ignore")

FILTERS    = ["r", "g"]    
APERTURE_R = 16            
SKY_IN     = 18            
SKY_OUT    = 26            
BOX_SIZE   = 15

STAR_COORDS = {
    ("20260316", "SZLYN")        : {"sz": (2188, 1673), "comp": [(1978, 1609), (2483, 1711), (1359, 1612), (1464, 1906)]},
    ("20260316", "SZLYN-2")      : {"sz": None,          "comp": []},
    ("20260414", "SZLYN")        : {"sz": None,          "comp": []},
    ("20260305", "SZ_Lyncis")    : {"sz": None,          "comp": []},
}

DATA_DIR = Path("data")
RED_DIR  = Path("reduced")
RED_DIR.mkdir(exist_ok=True)

NIGHTS = {
    "20260316": {
        "bias_glob": "bias-*.fit",
        "dark_glob": "Dark-*_60.fit",
        "flat_glob": {"r": "flat-*_r.fit",  "g": "flat-*_g.fit"},
        "runs": [
            {"label": "SZLYN",   "subdir": "szlyncis",  "science_glob": {"r": "SZLYN-[0-9]*_r60.fit", "g": "SZLYN-[0-9]*_g60.fit"}},
            {"label": "SZLYN-2", "subdir": "szlyncis2", "science_glob": {"r": "SZLYN-2-*_r60.fit",    "g": "SZLYN-2-*_g60.fit"}},
        ],
    },
    "20260414": {
        "bias_glob": "dark-*bias.fit",
        "dark_glob": "dark-*dark.fit",
        "flat_glob": {"r": "flat-*r.fit",   "g": "flat-*g.fit"},
        "runs": [
            {"label": "SZLYN", "subdir": "szlyncis", "science_glob": {"r": "sz_lyncis-*r.fit", "g": "sz_lyncis-*g.fit"}},
        ],
    },
    "20260305": {
        "bias_glob": "Calibration-*_bias.fit",
        "dark_glob": "Calibration-*_60s.fit",
        "flat_glob": {"r": "flat-*_r.fit",  "g": "flat-*_g.fit"},
        "runs": [
            {"label": "SZ_Lyncis", "subdir": "szlyncis", "science_glob": {"r": "SZ_Lyncis-*_r.fit", "g": "SZ_Lyncis-*_g.fit"}},
        ],
    },
}

def inv_median(a):
    return 1 / np.median(a)

def update_centroids(data, current_positions, box_size):
    new_positions = []
    
    for (x, y) in current_positions:
        x_int, y_int = int(round(x)), int(round(y))
        
        y_min, y_max = max(0, y_int - box_size), min(data.shape[0], y_int + box_size)
        x_min, x_max = max(0, x_int - box_size), min(data.shape[1], x_int + box_size)
        
        cutout = data[y_min:y_max, x_min:x_max]
        
        local_bkg = np.median(cutout)
        cutout_clean = np.maximum(cutout - local_bkg, 0)
        
        if np.sum(cutout_clean) > 0:
            xc_cutout, yc_cutout = centroid_com(cutout_clean)
            x_new = x_min + xc_cutout
            y_new = y_min + yc_cutout
            new_positions.append([x_new, y_new])
        else:
            new_positions.append([x, y])
            
    return np.array(new_positions, dtype=float)

def do_photometry(frames, night, run_label, filt, sz_xy, comp_stars):
    current_positions = np.array([sz_xy] + list(comp_stars), dtype=float)
    rows = []
    posities_historie = [] 

    for idx, (fname, ccd, hdr) in enumerate(frames):
        data = ccd.data
        current_positions = update_centroids(data, current_positions, box_size=BOX_SIZE)
        posities_historie.append(current_positions.copy())

        apertures   = CircularAperture(current_positions, r=APERTURE_R)
        annuli      = CircularAnnulus(current_positions, r_in=SKY_IN, r_out=SKY_OUT)
        
        if idx in [0, len(frames) // 2, len(frames) - 1]:
            _, ax = plt.subplots(figsize=(6, 6))
            vmin, vmax = np.percentile(data, [1, 99])
            ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
            
            apertures.plot(ax=ax, color='cyan', lw=1.5)
            
            ax.text(current_positions[0][0] + 15, current_positions[0][1], "SZ Lyn", color='yellow', fontsize=9, weight='bold')
            for c_idx, pos in enumerate(current_positions[1:]):
                ax.text(pos[0] + 15, pos[1], f"Comp {c_idx+1}", color='lawngreen', fontsize=8)
            
            sz_x, sz_y = current_positions[0]
            ax.set_xlim(sz_x - 40, sz_x + 40)
            ax.set_ylim(sz_y - 40, sz_y + 40)
            
            ax.set_title(f"Visual Tracking Check — Frame {idx}\n{fname}")
            plt.tight_layout()
            check_name = f"tracking_check_{night}_{run_label}_frame_{idx}.png".replace(" ", "_")
            plt.savefig(check_name, dpi=120)
            plt.close()
            print(f"  [Check] Visuele controle opgeslagen voor frame {idx} ({check_name})")

        phot        = aperture_photometry(data, apertures)
        sky         = aperture_photometry(data, annuli)
        
        #solving issue #3
        sky_per_pix = sky['aperture_sum'] / annuli.area
        net_flux    = phot['aperture_sum'] - sky_per_pix * apertures.area

        inst_mag  = -2.5 * np.log10(np.maximum(net_flux.value, 1.0))
        comp_mean = float(np.mean(inst_mag[1:])) 
        diff_mag  = float(inst_mag[0]) - comp_mean 

        hjd = hdr.get('JD-HELIO', hdr.get('JD', 0.0))
        rows.append({'filename': fname, 'HJD': round(hjd, 7), 'diff_mag': round(diff_mag, 5)})
        print(f"  {fname}  HJD={hjd:.6f}  diff_mag={diff_mag:+.4f}")

    if posities_historie:
        posities_historie = np.array(posities_historie) 
        frames_as = np.arange(len(frames))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        for s_idx in range(posities_historie.shape[1]):
            label = "SZ Lyn" if s_idx == 0 else f"Comparison {s_idx}"
            color = "steelblue" if s_idx == 0 else None 
            
            drift_x = posities_historie[:, s_idx, 0] - posities_historie[0, s_idx, 0]
            drift_y = posities_historie[:, s_idx, 1] - posities_historie[0, s_idx, 1]
            
            ax1.plot(frames_as, drift_x, label=label, color=color, alpha=0.8, lw=1.5)
            ax2.plot(frames_as, drift_y, label=label, color=color, alpha=0.8, lw=1.5)
            
        ax1.set_ylabel("Drift in X (pixels)")
        ax1.set_title("Wiskundige Tracking Controle: Lijnen MOETEN parallel lopen!")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", fontsize=8)
        
        ax2.set_ylabel("Drift in Y (pixels)")
        ax2.set_xlabel("Frame nummer")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        tag = f"{night}_{run_label}_{filt}".replace(" ", "_")
        drift_name = f"drift_check_{tag}.png"
        plt.savefig(drift_name, dpi=150)
        plt.close()
        print(f"  [Check] Wiskundige drift-plot opgeslagen: {drift_name}")

    tag       = f"{night}_{run_label}_{filt}".replace(" ", "_")
    csv_name  = f"lightcurve_{tag}.csv"
    plot_name = f"lightcurve_{tag}.png"

    with open(csv_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'HJD', 'diff_mag'])
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        return

    hjds  = [r['HJD'] for r in rows]
    mags  = [r['diff_mag'] for r in rows]
    hours = [(t - hjds[0]) * 24 for t in hjds]

    _, ax = plt.subplots(figsize=(10, 5))
    ax.invert_yaxis()
    ax.scatter(hours, mags, s=60, color='steelblue', zorder=3)
    ax.plot(hours, mags, color='steelblue', alpha=0.5)
    ax.set_xlabel(f"Hours since HJD {hjds[0]:.4f}")
    ax.set_ylabel("Differential magnitude (SZ Lyn − comp ensemble)")
    ax.set_title(f"SZ Lyncis — {night} / {run_label} — {filt} filter")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_name, dpi=150)
    plt.close()
    print(f"  Saved: {csv_name}  {plot_name}")

all_calibrated = []

for night, cfg in NIGHTS.items():
    cal_path = DATA_DIR / night / "calibration"
    red_path = RED_DIR  / night
    red_path.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Night: {night}")
    print(f"{'='*60}")
    
    print(f"\n--- § 2.4  Combining bias images ---")
    bias_files = sorted(cal_path.glob(cfg["bias_glob"]))
    if not bias_files:
        print(f"  WARNING: no bias files — skipping {night}")
        continue

    combined_bias = ccdp.combine(
        [str(f) for f in bias_files],
        method='average', sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6, unit='adu'
    )
    combined_bias.meta['combined'] = True
    combined_bias.uncertainty = None
    combined_bias.write(red_path / 'combined_bias.fit', overwrite=True)
    print(f"  {len(bias_files)} bias frames -> combined_bias.fit")
    
    print(f"\n--- § 3.6  Calibrating and combining dark frames ---")
    dark_files = sorted(cal_path.glob(cfg["dark_glob"]))
    if not dark_files:
        print(f"  WARNING: no dark files — skipping {night}")
        continue

    combined_dark = ccdp.combine(
        [ccdp.subtract_bias(CCDData.read(str(f), unit='adu'), combined_bias) for f in dark_files],
        method='average', sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6
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
            continue

        print(f"\n--- § 5.3 / 5.4  Combining flat frames (filter: {filt}) ---")
        calibrated_flats = []
        #Solving issue #1
        for f in flat_files:
            ccd = CCDData.read(str(f), unit='adu')
            ccd = ccdp.subtract_bias(ccd, combined_bias)
            ccd = ccdp.subtract_dark(ccd, combined_dark, exposure_time='EXPTIME', exposure_unit=u.second, scale=True)
            calibrated_flats.append(ccd)

        combined_flat = ccdp.combine(
            calibrated_flats, method='average', scale=inv_median,
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6
        )
        combined_flat.meta['combined'] = True
        combined_flat.uncertainty = None  
        combined_flat.write(red_path / f'combined_flat_{filt}.fit', overwrite=True)
        combined_flats[filt] = combined_flat
        print(f"  {len(flat_files)} flat frames -> combined_flat_{filt}.fit")
        
        for run in cfg["runs"]:
            sci_glob = run["science_glob"].get(filt)
            if not sci_glob: continue
            sci_files = sorted((DATA_DIR / night / run["subdir"]).glob(sci_glob))
            if not sci_files: continue

            print(f"\n--- § 6.3  Calibrating science images — {night} / {run['label']} / {filt} ---")
            run_red = red_path / run["subdir"]
            run_red.mkdir(exist_ok=True)

            #solving issue #1
            for f in sci_files:
                ccd = CCDData.read(str(f), unit='adu')
                reduced = ccdp.subtract_bias(ccd, combined_bias)
                reduced = ccdp.subtract_dark(reduced, combined_dark, exposure_time='EXPTIME', exposure_unit=u.second, scale=True)
                reduced = ccdp.flat_correct(reduced, combined_flat)
                reduced.write(run_red / f.name, overwrite=True)
                all_calibrated.append((night, run["label"], filt, f.name, reduced, ccd.header))
                print(f"  Calibrated: {f.name}")

if not all_calibrated:
    print("\nNo calibrated frames produced — check file paths.")
else:
    print("\n\n=== Generating field maps (one per run) ===")
    seen_runs = set()
    for night, run_label, filt, fname, ccd, hdr in all_calibrated:
        key = (night, run_label)
        if key in seen_runs: continue
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
            x_col = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
            y_col = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
            
            ax.scatter(sources[x_col], sources[y_col], s=40, facecolors='none', edgecolors='cyan', linewidths=0.8)
            for s in sources:
                ax.text(s[x_col] + 15, s[y_col], f"({s[x_col]:.0f}, {s[y_col]:.0f})", color='yellow', fontsize=5)
        ax.set_title(f"{night} / {run_label} — filter {filt}\nFind SZ Lyncis and comparison stars, note their (x, y) coordinates")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  {out}  ({len(sources) if sources else 0} stars detected)")

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
