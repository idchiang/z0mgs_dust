import os
import gzip
import shutil
from astropy.io import fits


def save_fits_gz(fn, data, hdr):
    if os.path.isfile(fn + '.fits.gz'):
        os.remove(fn + '.fits.gz')
    if data.dtype == bool:
        fits.writeto(fn, data.astype(int), hdr, overwrite=True)
    else:
        fits.writeto(fn, data, hdr, overwrite=True)
    # os.system("gzip -f " + fn)
    with open(fn, 'rb') as f_in:
        with gzip.open(fn + '.fits.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(fn)
