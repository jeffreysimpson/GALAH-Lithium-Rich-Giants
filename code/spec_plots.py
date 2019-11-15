import getpass
import os
import shutil
import subprocess

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy import constants as const
from astropy.io import fits
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')
username = getpass.getuser()
if username == "z3526655":
    basest_dir = f"/srv/scratch/z3526655/galah"
else:
    basest_dir = f"/Users/{username}/datacentral"

# Create the selection cuts


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


def spec_plotting(ax, star, camera, line_window, kwargs, need_tar):
    v_t = star['rv_guess'] * u.km / u.s
    v_b = 0. * u.km / u.s  # star['v_bary']
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
#     print(specfile)
    if not os.path.isfile(specfile):
        tar_name = f"{tar_dir}/{sobject_id[:6]}/standard/{com}.tar.gz"
    #     print(tar_name)
        if os.path.isfile(tar_name):
            tar_command = f"tar -xvzf {tar_name} */{sobject_id}{camera}.fits "
            print(f"Extracting: {specfile}")
            print(tar_command)
            cp = subprocess.run(
                tar_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            print(cp.stdout)
            untarred_file_location = cp.stdout[:-1]
            if not os.path.exists(new_file_dir):
                os.makedirs(new_file_dir)
            try:
                print(f"current: {untarred_file_location}")
                print(f"wanted: {specfile}")
                shutil.move(untarred_file_location, specfile)
            except FileNotFoundError:
                print("No file to move?!")
                return need_tar
        else:
            need_tar.add(tar_name.split("/")[8])
            return need_tar
    with fits.open(specfile) as spec:
        wavelength = (
            (spec[0].header['CDELT1'] *
             (np.arange(len(spec[0].data)) + 1 - spec[0].header['CRPIX1']) +
             spec[0].header['CRVAL1']))
        plot_range = [line_window["plot_centres"][0] - line_window['range'],
                      line_window["plot_centres"][0] + line_window['range']]
        median_idx = (
            wavelength > plot_range[0] -
            0) & (
            wavelength < plot_range[1] +
            0)
        ax.plot((wavelength * rv_offset)[median_idx],
                1.02*(spec[0].data / np.nanpercentile(spec[0].data[median_idx], 90))[median_idx],
                **kwargs)
    return need_tar


galah_dr3 = fits.open(f"{basest_dir}/GALAH_iDR3_main_alpha_190529.fits")[1].data

selection_idx, temp_grav_idx, li_selection_idx, li_rich_idx = selection_cuts(galah_dr3)

galah_median_dir = f"{basest_dir}/GALAH_local/GALAH_median_spectra"

line_windows = [{"element": "Li",
                 "plot_centres": [6708],
                 "range": 20,
                 "camera": [3]}]

need_tar = set()
star_num = 2
for star in galah_dr3[li_rich_idx][0:star_num+1]:
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
    need_tar = spec_plotting(axes, star, 3, line_windows[0], dict(lw=1, c='k'), need_tar)
    try:
        spec_list = os.listdir(median_spec_dir)
        useful_specs = [spec for spec in spec_list if spec.endswith("_3.csv")]
        for useful_spec in sorted(useful_specs):
            spec_open = pd.read_csv(f"{median_spec_dir}/{useful_spec}",
                                    comment='#', names=['wave', 'flux'])
            axes.plot(spec_open['wave'], spec_open['flux'], alpha=0.3, label=useful_spec[8:11])
    except FileNotFoundError:
        teff_idx = (galah_dr3['teff'] > star['teff']-50) & (galah_dr3['teff'] < star['teff']+50)
        logg_idx = (galah_dr3['logg'] > star['logg']-0.2) & (galah_dr3['logg'] < star['logg']+0.2)
        feh_idx = (galah_dr3['fe_h'] > star['fe_h']-0.5) & (galah_dr3['fe_h'] < star['fe_h']+0.5)
        snr_idx = galah_dr3["snr_c2_iraf"] > 100
        temp_grav_selection = galah_dr3[selection_idx & teff_idx & logg_idx & feh_idx & snr_idx]
        for test_star in temp_grav_selection[np.argsort(temp_grav_selection["snr_c2_iraf"])[::-1]][0:10]:
            need_tar = spec_plotting(axes, test_star, 3, line_windows[0], dict(alpha=0.3), need_tar)

    for line in [6703.576, 6705.105, 6707.76, 6707.98, 6710.323, 6713.044]:
        axes.axvspan(line-0.05, line+0.05, alpha=0.1, color='k')
    title_str = f"{star['sobject_id']}"
    for extra_str in [f" T$=${teff_round:0.0f}",
                      f" $\log g={logg_round:0.1f}$",
                      f" [Fe/H]$={fe_h_round:0.1f}$",
                      f" A_Li$={star['a_li']:0.2f}$"]:
        title_str += extra_str
    axes.legend()
    axes.set_title(title_str)
    axes.set_xlim(6708-20, 6708+20)
    axes.set_ylim(0, 1.1)
    axes.set_xlabel("Wavelength ($\AA$)")
    axes.set_ylabel("Normalized flux")
    plt.tight_layout()
    plt.savefig(f"spec_plots/spec_{star['sobject_id']}.pdf",
                bbox_inches='tight')
    plt.close('all')
#     break
print(need_tar)
