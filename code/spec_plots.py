import argparse
import getpass
import logging
import os
import shutil
import subprocess
import sys
from os.path import basename

import astropy.units as u
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy import constants as const
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from IPython.display import set_matplotlib_formats

# Create kernel
g = Gaussian1DKernel(stddev=150)

set_matplotlib_formats('retina')
username = getpass.getuser()
if username == "z3526655":
    basest_dir = f"/srv/scratch/z3526655/galah"
else:
    basest_dir = f"/Users/{username}/datacentral"


def selection_cuts(table):
    with np.errstate(invalid='ignore'):
        selection_idx = (table['flag_sp'] == 0) & (table['flag_fe_h'] == 0)
        temp_grav_idx = (table['teff'] > 4000) & (table['teff'] < 6000) & (
            table['logg'] > -1) & (table['logg'] < 3)
        li_selection_idx = selection_idx & temp_grav_idx & (
            np.in1d(table['flag_li'], [0, 1])) & (table['mass'] < 2.5)
        li_rich_idx = ((li_selection_idx) &
                       (table['a_li'] > 1.7) &
                       (table['flag_li'] == 0))
    return selection_idx, temp_grav_idx, li_selection_idx, li_rich_idx


def spec_plotting(ax, star, camera, line_window, kwargs, need_tar, offset=0.95):
    v_t = star['rv_guess'] * u.km / u.s
    # v_b = 0. * u.km / u.s  # star['v_bary']
    rv_offset = 1 / ((v_t) / const.c + 1)
    camera = 3
    sobject_id = str(star['sobject_id'])
    com = "com"
    if sobject_id[11] == "2":
        com += "2"
    tar_dir = f"{basest_dir}/GALAH/obs/reductions/Iraf_5.3"
    fits_dir = f"{basest_dir}/GALAH_local/obs/reductions/Iraf_5.3"
    specfile = f"{fits_dir}/{sobject_id[:6]}/{com}/{sobject_id}{camera}.fits"
    new_file_dir = f"{fits_dir}/{sobject_id[:6]}/{com}"
    if not os.path.isfile(specfile):
        tar_name = f"{tar_dir}/{sobject_id[:6]}/standard/{com}.tar.gz"
        # if username == "z3526655":
            # logging.info(f"Copying the file to local scratch: {local_scratch}")
            # tar_name = f"{tar_dir}/{sobject_id[:6]}/standard/{com}.tar.gz"
            # shutil.copy(tar_name, f'{local_scratch}/{tar_name.split("/")[-1]}')
            # tar_name = f'{local_scratch}/{tar_name.split("/")[-1]}'
        if os.path.isfile(tar_name):
            tar_command = f"tar -xvzf {tar_name} */{sobject_id}{camera}.fits "
            logging.info(f"Need to extract: {specfile}")
            logging.info(f"Using this tar command: {tar_command}")
            cp = subprocess.run(
                tar_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            #  for some reason katana is different?!
            if username == "z3526655":
                logging.info(f"Where file ended up: {cp.stdout}")
                untarred_file_location = cp.stdout[:-1]
            else:
                logging.info(f"Where file ended up: {cp.stderr}")
                untarred_file_location = cp.stderr[2:-1]
            if not os.path.exists(new_file_dir):
                os.makedirs(new_file_dir)
            try:
                logging.info(f"Current location: {untarred_file_location}")
                logging.info(f"New location: {specfile}")
                shutil.move(untarred_file_location, specfile)
            except FileNotFoundError:
                logging.info("No file to move?!")
                return need_tar
        else:
            need_tar.add(tar_name.split("/")[8])
            return need_tar
    logging.info(f"Ploting: {specfile}")
    with fits.open(specfile) as spec:
        wavelength = (
            (spec[0].header['CDELT1'] *
             (np.arange(len(spec[0].data)) + 1 - spec[0].header['CRPIX1']) +
             spec[0].header['CRVAL1']))
        # Convolve data
        z = convolve(spec[0].data, g)
        normed_spec = (spec[0].data)/z
        percentile_region_idx = (wavelength > 6708-10) & (wavelength < 6708+10)
        percentile_norm = np.percentile(normed_spec[percentile_region_idx], 99)
        ax.plot(wavelength * rv_offset,
                normed_spec/percentile_norm,
                **kwargs)
    return need_tar

def near_spectra(star, need_tar):
    with np.errstate(invalid='ignore'):
        teff_idx = (galah_dr3['teff'] > star['teff']-50) & (galah_dr3['teff'] < star['teff']+50)
        logg_idx = (galah_dr3['logg'] > star['logg']-0.2) & (galah_dr3['logg'] < star['logg']+0.2)
        feh_idx = (galah_dr3['fe_h'] > star['fe_h']-0.2) & (galah_dr3['fe_h'] < star['fe_h']+0.2)
        snr_idx = galah_dr3["snr_c2_iraf"] > 100
        no_same_star_idx = ~np.in1d(galah_dr3['source_id'], star['source_id'])
    temp_grav_selection = galah_dr3[selection_idx & teff_idx & logg_idx & feh_idx & snr_idx & no_same_star_idx]
    for test_star in temp_grav_selection[np.argsort(temp_grav_selection["snr_c2_iraf"])[::-1]][0:10]:
        need_tar = spec_plotting(axes, test_star, 3, line_windows[0],
                                 dict(lw=0.5, alpha=0.6, c=m.to_rgba(star['fe_h'])), need_tar)
    return need_tar


parser = argparse.ArgumentParser(
    description="Create the lithium spec plot. Only provide one of the arguments.",
    usage=f"{basename(__file__)} -sobject_id <sobject_id> -index_num <index_number>")

parser.add_argument('-sobject_id',
                    required=False,
                    type=int,
                    help="sobject_id to plot")
parser.add_argument('-index_num',
                    required=False,
                    type=int,
                    help="index_num to plot")
parser.add_argument('--log_to_screen', dest='log', action='store_true')
parser.add_argument('--log_to_file', dest='log', action='store_false')
parser.set_defaults(log=True)

if len(sys.argv[1:]) in [0, 4]:
    print()
    parser.print_help()
    parser.exit()
    print()

args = parser.parse_args()
sobject_id = args.sobject_id
index_num = args.index_num
LOG_TO_SCREEN = args.log

galah_dr3 = fits.open(f"{basest_dir}/GALAH_iDR3_main_alpha_190529.fits")[1].data

(selection_idx, temp_grav_idx,
 li_selection_idx, li_rich_idx) = selection_cuts(galah_dr3)

galah_median_dir = f"{basest_dir}/GALAH_local/GALAH_median_spectra"

line_windows = [{"element": "Li",
                 "plot_centres": [6708],
                 "range": 20,
                 "camera": [3]}]

if LOG_TO_SCREEN:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s')
else:
    logging.basicConfig(
        filename=f'logs/output_{sobject_id}_{index_num}.log',
        filemode='w', level=logging.INFO,
        format='%(asctime)s - %(message)s')

if sobject_id is not None:
    try:
        star = galah_dr3[galah_dr3['sobject_id'] == sobject_id][0]
    except IndexError:
        logging.info(f"Not valid sobject_id")
        sys.exit()
if index_num is not None:
    other_stars_idx = li_selection_idx & ~li_rich_idx
    other_stars_idx[other_stars_idx.cumsum() > 500] = False
    try:
        star = galah_dr3[other_stars_idx | li_rich_idx][index_num]
        sobject_id = star['sobject_id']
    except IndexError:
        logging.info(f"Index out of range. Largest value is {np.sum(li_rich_idx)-1}")
        sys.exit()


norm = mpl.colors.Normalize(vmin=-1, vmax=0.2)
cmap = cm.viridis_r
m = cm.ScalarMappable(norm=norm, cmap=cmap)

need_tar = set()

# for star in galah_dr3[li_rich_idx][160:165+1]:
logging.info(f"Starting sobject_id {sobject_id}")
with np.errstate(invalid='ignore'):
    teff_round = np.round(star['teff']/50)*50
    logg_round = np.round(star['logg']/0.2)*0.2
    fe_h_round = np.round(star['fe_h']/0.1)*0.1
fe_h_str = f"{fe_h_round:+0.1f}".replace("+", "p").replace("-", "m").replace(".", "")
median_spec_dir = f"{galah_median_dir}/T{teff_round:0.0f}/g{logg_round*10:0.0f}"

sns.set_context("paper", font_scale=1.2)
fig, axes = plt.subplots(nrows=1,
                         ncols=1,
                         figsize=(10, 5), sharex='col', sharey='col')
need_tar = spec_plotting(axes, star, 3,
                         line_windows[0], dict(lw=1, c='k'), need_tar, offset=0.94)
# try:
#     spec_list = os.listdir(median_spec_dir)
#     useful_specs = [spec for spec in spec_list if spec.endswith("_3.csv")]
#     for useful_spec in sorted(useful_specs):
#         spec_open = pd.read_csv(f"{median_spec_dir}/{useful_spec}",
#                                 comment='#', names=['wave', 'flux'])
#         axes.plot(spec_open['wave'],
#                   spec_open['flux'],
#                   c=m.to_rgba(float(useful_spec[8:11].replace('p', '+').replace('m', '-'))/10),
#                   alpha=0.6,
#                   lw=0.5,
#                   label=useful_spec[8:11])
# except FileNotFoundError:
need_tar = near_spectra(star, need_tar)

for line in [6703.576, 6705.105, 6707.76, 6710.323, 6713.044]: #6707.98,
    axes.axvspan(line-0.05, line+0.05, alpha=0.1, color='k')
title_str = f"{star['sobject_id']}"
for extra_str in [f" T$=${teff_round:0.0f}",
                  f" $\log g={logg_round:0.1f}$",
                  f" [Fe/H]$={fe_h_round:0.1f}$",
                  f" A_Li$={star['a_li']:0.2f}$"]:
    title_str += extra_str
# axes.legend()
axes.set_title(title_str)
axes.set_xlim(6708-10, 6708+10)
axes.set_ylim(0, 1.1)
axes.set_xlabel(r"Wavelength ($\AA$)")
axes.set_ylabel("Normalized flux")
#    cbar = plt.colorbar(m)
plt.tight_layout()
plt.savefig(f"spec_plots/spec_{star['sobject_id']}.pdf",
            bbox_inches='tight')
plt.close('all')
#     break
logging.info(need_tar)
