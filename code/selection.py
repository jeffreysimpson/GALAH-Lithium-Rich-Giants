

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
