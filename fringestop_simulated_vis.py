import numpy as np
from scipy.constants import c
from caput.astro.coordinates import spherical

# draco -> interferometry.py
def fringestop_phase(ha, lat, dec, u, v, w=0.0):
    """Return the phase required to fringestop. All angle inputs are radians.

    Note that for a visibility V_{ij} = < E_i E_j^*>, this expects the u, v,
    w coordinates are the components of (d_i - d_j) / lambda.

    Parameters
    ----------
    ha : array_like
        The Hour Angle of the source to fringestop too.
    lat : array_like
        The latitude of the observatory.
    dec : array_like
        The declination of the source.
    u : array_like
        The EW separation in wavelengths (increases to the E)
    v : array_like
        The NS separation in wavelengths (increases to the N)
    w : array_like, optional
        The vertical separation on wavelengths (increases to the sky!)

    Returns
    -------
    phase : np.ndarray
        The phase required to *correct* the fringeing. Shape is
        given by the broadcast of the arguments together.
    """
    phase = -2.0j * np.pi * spherical.projected_distance(ha, lat, dec, u, v, w)

    return np.exp(phase, out=phase)

# draco -> draco -> analysis -> beam.py -> CreateBeamStreamFromTelescope.process
# Determine baseline distances
lmbda = c * 1e-6 / freq
u = x[np.newaxis, :] / lmbda[:, np.newaxis]
u = u[:, :, np.newaxis, np.newaxis]

# "Rotate the baseline distances by the telescope's rotation angle.
# This assumes that the baseline distances used to beamform in the
# NS direction were NOT rotated, and hence the phase due to that rotation
# should be corrected by the beam.  Note that we can only partially
# correct for the rotation in this way, since we have already collapsed
# over NS baselines. However, this partial correction should be pretty good
# for small rotation angles and for sources near meridian."
rot = np.radians(self.telescope.rotation_angle)
v = np.sin(rot) * u
u = np.cos(rot) * u

# Calculate the phase
phi = fringestop_phase( arr_ha, np.radians(self.telescope.latitude), arr_dec, 
                       u, v 
                       ).conj()