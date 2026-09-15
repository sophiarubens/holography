import numpy as np
from matplotlib import pyplot as plt

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity
from astropy import units as u

import fftvis
import matvis

import healpy as hp

from pyuvdata.analytic_beam import AiryBeam,GaussianBeam #,UniformBeam

from scipy.fft import fftfreq,fftshift

NFREQS=20
FREQS=np.linspace(590,610,NFREQS)*u.MHz
NTIMES=30
JD0=2461274
JD1=JD0+0.05
TIMES=Time(np.linspace(JD0, JD1, NTIMES), format="jd", scale="utc") 
DRAO=EarthLocation.of_site("drao") # coordinates in the astropy database are for Galt, but this is convenient and good enough for now

# fftvis convention:
# polarized visibility matrices are indexed as (N freqs, N times, N feed i, N feed j, N baselines)
# unpolarized visibility matrices are indexed as (N freqs, N times, N antennas, N antennas) *IMPORTANT* AnalyticBeam objects don't support polarization

# based upon fftvis_tutorial.ipynb

# sky model (use a point source decomposition and healpix representation)

# RAs are just placeholders for now because I'm tired of tracking down catalogue values for today
CHIME_match={"CygA": {"dec":40.7, "ra":0,    "S600":3613, "alpha":-0.82, "confN": 0.02}, # plausible values not in direct tension with https://www.aanda.org/articles/aa/full_html/2020/03/aa36844-19/aa36844-19.html#S8
             "CasA": {"dec":58.8, "ra":2,    "S600":2375, "alpha":-2,    "confN": 0.03},
             "TauA": {"dec":22.0, "ra":4,    "S600":1142, "alpha":-0.3,  "confN": 0.06},
             "PerB": {"dec":29.7, "ra":6,    "S600":  87, "alpha":-1.3,  "confN": 0.8}, # https://arxiv.org/html/2603.23587v1#S5
             "3C10C":{"dec":64.2, "ra":8,    "S600":  71, "alpha":-0.62, "confN": 7}, # couldn't find an actual source or any references other than the CHIME paper (although I didn't do a full recursive search) so I'm using the 3C10 value as a placeholder
             "3C84": {"dec":41.5, "ra":10,   "S600":  38, "alpha":-0.93, "confN":10}, # from the 3C catalogue paper
             "3C295":{"dec":52.5, "ra":12,   "S600":  37, "alpha":-0.08, "confN": 2},
             "3C58": {"dec":64.8, "ra":14,   "S600":  31, "alpha":+0.11, "confN": 2},
             "3C147":{"dec":49.9, "ra":16,   "S600":  29, "alpha":+0.77, "confN": 2},
             "3C111":{"dec":38.0, "ra":18,   "S600":  29, "alpha":-0.75, "confN": 2},
             "3C196":{"dec":48.2, "ra":20,   "S600":  28, "alpha":-0.68, "confN": 2},
             "3C409":{"dec":23.6, "ra":22,   "S600":  27, "alpha":-0.78, "confN": 3},
             "3C48": {"dec":33.2, "ra":23,   "S600":  25, "alpha":-0.07, "confN": 3},
             "3C286":{"dec":30.5, "ra":23.5, "S600":  18, "alpha":-0.19, "confN": 4}
            } # radio point sources from the CHIME 2024 holography paper that could appear at boresight for CHORD

# similar to simulate_sky but just the bright catalogue sources
CHIME_RA=[elem.ra for elem in CHIME_match.values()]
CHIME_DEC=[elem.dec for elem in CHIME_match.values()]
CHIME_ALPHAS=[elem.alpha for elem in CHIME_match.values()]
CHIME_S600=[elem.S600 for elem in CHIME_match.values()]
CHIME_FLUX_ALLFREQ=((FREQS[:, np.newaxis] / FREQS[0]) ** CHIME_ALPHAS.T * CHIME_S600.T).T

# CHORD layout
CHORD_NS_bl=8.5*u.m
CHORD_EW_bl=6.3*u.m
CHORD_N_NS=24
CHORD_N_EW=22

CHORD_NS_vec=CHORD_NS_bl*fftfreq(fftshift(CHORD_N_NS))
CHORD_EW_vec=CHORD_EW_bl*fftfreq(fftshift(CHORD_N_EW))
CHORD_EW_coords,CHORD_NS_coords=np.meshgrid(CHORD_EW_vec,CHORD_NS_vec,indexing="ij")
Galt_EW=CHORD_EW_vec[-1,-1]+116*u.m
Galt_NS=CHORD_NS_vec[-1,-1]+  6*u.m
print("CHORD_EW_coords.shape=",CHORD_EW_coords.shape)
CHORD_ant_mask=np.ones((24,22),dtype="bool")
CHORD_ant_mask[   0 ,   0  ]=False # NW corner observatory access road gap
CHORD_ant_mask[   6 ,  10  ]=False # N receiver hut
CHORD_ant_mask[  17 ,  10  ]=False # S receiver hut
CHORD_ant_mask[ - 6:, - 2: ]=False # SE corner too steep for antennas
CHORD_ant_mask[  17 , - 1  ]=False # extra antenna missing from SE corner
CHORD_EW=CHORD_EW_coords[CHORD_ant_mask]
CHORD_NS=CHORD_NS_coords[CHORD_ant_mask]
holog_EW=CHORD_EW.append(Galt_EW) # should be a list by now
holog_NS=CHORD_NS.append(Galt_NS)
# clicking 116 m east, 6 m north

Nmini=2
mini_array_EW, mini_array_NS = np.meshgrid(CHORD_NS_bl*fftshift(fftfreq(Nmini)),
                                           CHORD_EW_bl*fftshift(fftfreq(Nmini)), indexing="ij")
holog_mini_EW=list(np.reshape(mini_array_EW,Nmini**2)).append(Galt_EW)
holog_mini_NS=list(np.reshape(mini_array_NS,Nmini**2)).append(Galt_NS)

def coord_arrays_to_HERA_format(x_arr,y_arr,z_arr=None,antenna_mask=None):
    if z_arr is None:               # fallback: flat array
        z_arr=np.zeros_like(x_arr)
    if isinstance (x_arr,Quantity): # strip astropy units
        x_arr=x_arr.value
        y_arr=y_arr.value
        z_arr=z_arr.value

    N_ant=np.size(z_arr)
    x_flattened=np.reshape(x_arr,(N_ant,)) # make 1D
    y_flattened=np.reshape(y_arr,(N_ant,))
    z_flattened=np.reshape(z_arr,(N_ant,))

    if antenna_mask is not None:
        mask_flattened=np.reshape(antenna_mask,(N_ant,))
        x_flattened=x_flattened[mask_flattened]
        y_flattened=y_flattened[mask_flattened]
        z_flattened=z_flattened[mask_flattened]

    coords_all=np.asarray([x_flattened,y_flattened,z_flattened])
    coords_HERA_format=dict(enumerate(coords_all))
    return coords_HERA_format 

def simulate_sky(Nside=64,
                 freqs=FREQS,
                 mode="CHIME_match"):
    Npix=hp.nside2npix(Nside)
    # sky coord grid is a precursor for populating pixels as point sources distributed randomly across the sky
    ra = np.deg2rad(np.random.uniform(0, 360, Npix))     # source RAs (rad)
    dec = np.deg2rad(np.random.uniform(-90, 90.0, Npix)) # source decs (rad)
    dec, ra = hp.pix2ang(Nside, np.arange(Npix))
    dec -= np.pi / 2

    # consider only smooth-spectrum sources
    flux = np.random.uniform(0, 1, Npix) # source fluxes at 100MHz (Jy)
    alpha = np.ones(Npix) * -0.8         # source spectral indices

    if mode=="CHIME_match": # overwrite some of the random sources with bright point sources from the dictionary
        for i,src in enumerate(CHIME_match):
            S600=src["S600"]
            alpha[i]=src["alpha"]
            flux[i]=S600*(1/6)**alpha # S100
            ra[i]=src["ra"]
            dec[i]=src["dec"]

    # spectra of all sources (Nsource x Nfreq) 
    spectra = ((freqs[:, np.newaxis] / freqs[0]) ** alpha.T * flux.T).T

    return [ra,dec],spectra
    
def simulate_visibilities(simulator="fftvis",
                          antpos=coord_arrays_to_HERA_format(holog_mini_EW,holog_mini_NS),
                          beam=AiryBeam(diameter=6.0),
                          Nfreqs=NFREQS,
                          freqs=np.linspace(300e6,1500e6,NFREQS),
                          Ntimes=NTIMES,
                          times=TIMES,
                          telescope_loc=DRAO):
    if simulator=="fftvis":
        visibilities=fftvis.simulate_vis(
                                          ants=antpos,
                                          fluxes=CHIME_FLUX_ALLFREQ,
                                          ra=CHIME_RA,
                                          dec=CHIME_DEC,
                                          freqs=FREQS,
                                          times=TIMES.jd,
                                          telescope_loc=telescope_loc,
                                          beam=beam,
                                          polarized=False,
                                          precision=2,
                                          nprocesses=1,
                                          baselines=baselines,
                                          backend="cpu"  # Explicitly specify the backend (new parameter)
                                        ) # cf. https://github.com/tyler-a-cox/fftvis/blob/main/docs/tutorials/fftvis_tutorial.ipynb
    elif simulator=="matvis":
        rng=np.random.default_rng()
        visibilities=matvis.simulate_vis(
                                          ants=antpos,
                                          fluxes=CHIME_FLUX_ALLFREQ,
                                          ra=CHIME_RA,
                                          dec=CHIME_DEC,
                                          freqs=FREQS,
                                          times=TIMES.jd,
                                          telescope_loc=telescope_loc,
                                          beams=[GaussianBeam(sigma=0.5),GaussianBeam(sigma=0.51)], beam_idx=rng.randint(len(antpos)),
                                          polarized=False,
                                          precision=2, # single/double precision, i.e. 32- vs. 64-bit floats
                                          # nprocesses=, # I'm pretty sure this only figures into the fftvis algorithm
                                          # baselines=,
                                          # backend= # I'm pretty sure this only figures into the fftvis algorithm
                                        ) # cf. https://matvis.readthedocs.io/en/latest/tutorials/matvis_tutorial.html
    else:
        raise ValueError("Unknown drift-scan visibility simulator. Try fftvis or matvis")
    return None

def extract_CHORD_x_Galt(N2:np.ndarray,baselines_with_CHORD,baselines_with_Galt):
    freq_axis=0 # I dunno if these will come in handy
    time_axis=1
    N2_ndim=N2.ndim # number of dimensions in the visibility matrix
    if N2_ndim==6: # Nfreq, Ntime, Nfeed, Nfeed, Nant, Nant
        feed_axes=[2,3]
    elif N2_ndim!=4: # Nfreq, Ntime, Nant, Nant
        raise TypeError("shape of visibility matrix is consistent with neither polarized nor unpolarized visibilities")

    CHORD_Galt_baselines=np.nonzero(baselines_with_CHORD & baselines_with_Galt)
    N2temp=      np.take_along_axis(N2,     CHORD_Galt_baselines, axis=-1)
    N2_filtered= np.take_along_axis(N2temp, CHORD_Galt_baselines, axis=-1)# original axis -2 is current axis -1
    return N2_filtered