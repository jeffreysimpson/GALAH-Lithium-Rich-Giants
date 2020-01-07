

# import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord  # EarthLocation,
# import seaborn as sns


def load_table(file_name, UV=False):
    with fits.open(file_name) as hdul:
        # This adds the absolute magnitude of the stars to the table
        orig_table = hdul[1].data
        orig_cols = orig_table.columns
        new_cols_list = [
            fits.Column(name='abs_G', format='D',
                        array=hdul[1].data['phot_g_mean_mag'] -
                        (5*np.log10(hdul[1].data['r_est'])-5)),
            fits.Column(name='j_ks', format='D',
                        array=hdul[1].data['j_m'] - hdul[1].data['ks_m'])]
        if UV:
            new_cols_list.append(fits.Column(
                name='FUV_NUV', format='D',
                array=hdul[1].data['FUVmag'] - hdul[1].data['NUVmag']))
        new_cols = fits.ColDefs(new_cols_list)
    hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols)
    galah_idr3 = hdu.data
    galah_idr3_position = SkyCoord(
        ra=galah_idr3['RA']*u.deg,
        dec=galah_idr3['DEC']*u.deg,
        frame='icrs')
    return galah_idr3, galah_idr3_position


def pmra_cut(table, pmra_0=0, pmdec_0=0, cut=1.5):
    # This does the cut on proper motion
    return ((table['pmra'] - pmra_0)**2 + (table['pmdec'] - pmdec_0)**2)**0.5 < cut


def pos_cut(positions, ra_0=78.0*u.deg, dec_0=-68.08*u.deg, radius=5*u.deg):
    # This does the cut on sky position
    centre = SkyCoord(ra=ra_0, dec=dec_0, frame='icrs')
    return centre.separation(positions) < radius


def create_selections(galah_idr3, galah_idr3_position):
    super_li = 2.7
    li_cut = 1.5
    li_rich_idx = galah_idr3['a_li'] > li_cut

    good_spec_idx = ((galah_idr3['flag_sp'] == 0) &
                 (galah_idr3['flag_fe_h'] == 0))
    li_measured_idx = ~np.isnan(galah_idr3['a_li'])

    giant_idx = ((galah_idr3['teff'] > 3000) &
                 (galah_idr3['teff'] < 5730) &
                 (galah_idr3['logg'] > -1) &
                 (galah_idr3['logg'] < 3))

    galactic_centre_idx = np.isin(galah_idr3['field_id'], [6702, 6738, 6736])

    lmc_idx = ((pmra_cut(galah_idr3, pmra_0=1.8, pmdec_0=0.25, cut=1.5) &
            (pos_cut(galah_idr3_position, ra_0=78.0*u.deg, dec_0=-68.08*u.deg, radius=5*u.deg)) &
            (galah_idr3['rv_guess'] > 215)))

    smc_idx = ((pmra_cut(galah_idr3, pmra_0=0.85, pmdec_0=-1.2, cut=1.5) &
            (pos_cut(galah_idr3_position, ra_0=11.83*u.deg, dec_0=-74.11*u.deg, radius=5*u.deg)) &
            (galah_idr3['rv_guess'] > 80)) &
            (galah_idr3['parallax'] < 0.08))

    too_red_hot_idx = (galah_idr3['teff'] > 4750) & (galah_idr3['j_m'] - galah_idr3['ks_m'] > 1.1)

    ignore_stars_idx = (good_spec_idx & giant_idx) & (galactic_centre_idx | lmc_idx | smc_idx | too_red_hot_idx) # All the stars we are ignoring

    # ignore_stars_li_normal_idx = ignore_stars_idx & ~li_rich_idx & li_measured_idx
    # ignore_stars_li_rich_idx = ignore_stars_idx & li_rich_idx & li_measured_idx
    #
    # everything_idx = good_spec_idx & ~giant_idx & ~ignore_stars_idx #Stars with good spec but not giants
    # good_giants_idx = good_spec_idx & giant_idx & ~ignore_stars_idx & ~li_rich_idx & li_measured_idx #Giants stars with good spec
    #
    # li_rich_giants_idx = good_spec_idx & giant_idx & ~ignore_stars_idx & li_rich_idx

    return ignore_stars_idx, good_spec_idx, giant_idx, li_measured_idx
