## Directory Structure

- `pipeline.py`: contains the main script of our research, this script runs the photometry and creates the lightcurves.
- `calibration.py`: contains solely the calibration part of our research, since this take some time we have decided to do this seperately. So the pipeline doesn't take as long when we decide to change things to the script (and it has to recalibrate again ect ect).
- `star_picker.py`: contains the script that chooses sz_lyn and 4 comparison stars from our FITS-files. These coordinates are saved in a .csv file and used in our pipeline.py for our photometry. 


