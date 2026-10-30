import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm,CenteredNorm,SymLogNorm

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity
from astropy import units as u

# from caput.astro.coordinates import spherical
import healpy as hp
import matvis

from pyuvdata.analytic_beam import AiryBeam
from pyuvdata import BeamInterface

from scipy.constants import c
from scipy.fft import fftfreq,fftshift
from scipy.stats import binned_statistic_2d

import time
import cmasher as cmr

# TODO: generalize to more realistic values, chunk up the band with different Nsides, etc. 
GLOBAL_NSIDE=512
freq_hi=320
freq_lo=300
FREQS=np.arange(freq_lo,freq_hi,0.195)*u.MHz
freq_range=freq_hi-freq_lo
NFREQS=len(FREQS)
JD0=2461274
JD1=JD0+0.25
time_vec=np.arange(JD0, JD1, 10/3600)
NTIMES=len(time_vec)
TIMES=Time(time_vec, format="jd", scale="utc") 
DRAO=EarthLocation.of_site("drao") # coordinates in the astropy database are for Galt, but this is convenient and good enough for now
omega_e=np.pi/12 # 2pi rad / 24 hr = pi/12
recalc = False
pointing_centre_source_name="CygA"

# polarized visibility matrices are indexed as (N freqs, N times, N feed i, N feed j, N baselines)
# unpolarized visibility matrices are indexed as (N freqs, N times, N antennas, N antennas) 
#    AnalyticBeam objects don't support polarization
#    in practice, the returned visibilities are actually (N freqs, N times, N baselines)

# caput -> caput -> astro -> coordinates -> spherical.py (I couldn't install caput locally since the wheels kept failing to build, even when I did an OpenMP override, and another 10-hr library installation adventure is not worth my time this week)
def projected_distance(ha, lat, dec, x, y, z=0.0):
    """Compute distance projected in the direction of a source.

    Parameters
    ----------
    ha : array_like
        The CIRS hour angle of the source in radians.
    lat : array_like
        The latitude of the observatory in radians.
    dec : array_like
        The CIRS declination of the source in radians.
    x : array_like
        The East-West coordinate in wavelengths (increases to the East).
    y : array_like
        The North-South coordinate in wavelengths (increases to the North).
    z : array_like
        The vertical coordinate in wavelengths (increases upwards).

    Returns
    -------
    distance : ndarray
        The projected distance. Has whatever units `x`, `y`, `z` had.
    """
    # We could use `sph_to_ground` here, but it's likely to be more memory
    # efficient to do this directly
    dist = x * (-1 * np.cos(dec) * np.sin(ha))
    dist += y * (np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(ha))
    dist += z * (np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha))

    return dist

# draco -> draco -> util -> interferometry.py -> fringestop_phase
def fringestop_phase_term(ha, lat, dec, u, v, w=0.0):
    """Return the phase required to fringestop. All angle inputs are radians.

    Note that for a visibility V_{ij} = < E_i E_j^*>, this expects the u, v,
    w coordinates are the components of (d_i - d_j) / lambda.

    Parameters
    ----------
    ha, dec : array_like
        Of the source 
    lat     : array_like
        Of the observatory
    u, v, w : array_like
        In lambdas
    Returns
    -------
    phase : np.ndarray
        The phase required to *correct* the fringing. Shape is
        given by the broadcast of the arguments together.
    """
    # print("ha, lat, dec, u, v, w =",ha, lat, dec, u, v, w)
    print("types of ha, lat, dec, u, v, w =",type(ha),type(lat),type(dec),type(u),type(v),type(w))
    phase = -2.0j * np.pi * projected_distance(ha, lat, dec, u, v, w)

    return np.exp(phase, out=phase)

# TODO: more rigorous fact-checking of the first references I found for these values
CHIME_match={"CygA": {"dec":40.7, "ra":19.9912, "S600":3613, "alpha":-0.82, "confN": 0.02}, # plausible based on https://www.aanda.org/articles/aa/full_html/2020/03/aa36844-19/aa36844-19.html#S8
             "CasA": {"dec":58.8, "ra":23.3900, "S600":2375, "alpha":-2,    "confN": 0.03},
             "TauA": {"dec":22.0, "ra":5.5755,  "S600":1142, "alpha":-0.3,  "confN": 0.06},
             "PerB": {"dec":29.7, "ra":4.6180,  "S600":  87, "alpha":-1.3,  "confN": 0.8}, # https://arxiv.org/html/2603.23587v1#S5
             "3C10C":{"dec":64.2, "ra":0.4203,  "S600":  71, "alpha":-0.62, "confN": 7}, # unclear what they meant by this -> currently using 3C10 value as placeholder
             "3C84": {"dec":41.5, "ra":3.3300,  "S600":  38, "alpha":-0.93, "confN":10}, # from the 3C catalogue paper
             "3C295":{"dec":52.5, "ra":14.1889, "S600":  37, "alpha":-0.08, "confN": 2},
             "3C58": {"dec":64.8, "ra":2.0936,  "S600":  31, "alpha":+0.11, "confN": 2},
             "3C147":{"dec":49.9, "ra":5.7101,  "S600":  29, "alpha":+0.77, "confN": 2},
             "3C111":{"dec":38.0, "ra":4.3058,  "S600":  29, "alpha":-0.75, "confN": 2},
             "3C196":{"dec":48.2, "ra":8.2267,  "S600":  28, "alpha":-0.68, "confN": 2},
             "3C409":{"dec":23.6, "ra":20.2410, "S600":  27, "alpha":-0.78, "confN": 3},
             "3C48": {"dec":33.2, "ra":1.6281,  "S600":  25, "alpha":-0.07, "confN": 3},
             "3C286":{"dec":30.5, "ra":13.5189, "S600":  18, "alpha":-0.19, "confN": 4}
            } # radio point sources from the CHIME 2024 holography paper that could appear at boresight for CHORD

def fringe_rate(H=12*u.hourangle, delta=0*u.hourangle, # source hr angle and dec (defaults to zenith for drift-scan)
                h=0*u.hourangle, d=49*u.deg,           # baseline-at-current-pointing hr angle and dec
                D=100*u.m,                             # baseline length
                obsfreq=600*u.MHz,                     # obs freq (default = 600 MHz)
                f_r_instr=0                            # instrumental clock contrib to fringe rate
                ):
    lambdaobs=3e8*u.m/u.s/obsfreq
    D_lambda=D/lambdaobs
    print("D_lambda=",D_lambda)
    f_r=-omega_e*D_lambda\
        *np.cos(d.to(u.rad))\
        *np.cos(delta.to(u.rad))\
        *np.sin((H-h).to(u.rad))\
        +f_r_instr
    return f_r.decompose()


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

    coords_all=np.asarray([x_flattened,y_flattened,z_flattened]).T
    coords_HERA_format=dict(enumerate(coords_all))
    return coords_HERA_format 

def simulate_sky(Nside=GLOBAL_NSIDE,
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
    
def simulate_visibilities(matvis_backend="cpu",
                          antpos=None,
                          freqs=np.linspace(300e6,1500e6,NFREQS),
                          times=TIMES,
                          telescope_loc=DRAO):
    leading_order_beams=[AiryBeam(diameter=6.0),AiryBeam(diameter=26.0)]
    antenna_diameter_indices=np.zeros(len(antpos),dtype=int)
    antenna_diameter_indices[-1]=int(1)
    use_gpu=True if matvis_backend=="gpu" else False
    visibilities=matvis.simulate_vis(ants=antpos,
                                    fluxes=CHIME_FLUX_ALLFREQ,
                                    ra=CHIME_RA,
                                    dec=CHIME_DEC,
                                    freqs=freqs,
                                    times=times,
                                    telescope_loc=telescope_loc,
                                    beams=leading_order_beams, beam_idx=antenna_diameter_indices,
                                    polarized=False,
                                    precision=2, # single/double precision, i.e. 32- vs. 64-bit floats
                                    # nprocesses=, # I'm pretty sure this only figures into the fftvis algorithm                                        # baselines=,
                                    use_gpu=use_gpu
                                    ) # cf. https://matvis.readthedocs.io/en/latest/tutorials/matvis_tutorial.html
    return visibilities

def manually_track_with_Galt(matvis_backend="cpu",
                             antpos=None,
                             freqs=np.linspace(300e6,1500e6,NFREQS),
                             leading_order_beams=None, antenna_diameter_indices=None,
                             time_vec=TIMES,
                             telescope_loc=DRAO,
                             Nside=GLOBAL_NSIDE):
    Nant=len(antpos)
    Ntime=len(time_vec)
    Nfreq=len(freqs) # might not be NFREQS in the case of having overwritten the keyword fallback
    Galt_tracked_visibilities=np.zeros((Nfreq,Ntime,Nant**2),dtype=np.complex128)

    if leading_order_beams is None:
        CHORD_beam=AiryBeam(diameter=26.)
        Galt_beam=AiryBeam(diameter=26.) # putting Galt last is literally just an arbitrary choice
        try:
            CHIME_match[str(pointing_centre_source_name)]
        except:
            print("pointing_centre_source_name=",pointing_centre_source_name)
            raise ValueError("invalid source name (absent from current catalogue)")

        
        leading_order_beams=[AiryBeam(diameter=6.),AiryBeam(diameter=26.)]
        antenna_diameter_indices=np.zeros(Nant,dtype=int)
        antenna_diameter_indices[-1]=int(1)
        CHORDbeam_idx,Galtbeam_idx=np.meshgrid(antenna_diameter_indices,antenna_diameter_indices,
                                               indexing="ij")
        heterogeneous_baseline_extraction_mask=(CHORDbeam_idx+Galtbeam_idx)==1 # the entries with values of 1 are the baselines to extract (0 = two CHORD beams; 2 = two Galt beams)
        heterogeneous_baseline_extraction_mask=np.asarray(heterogeneous_baseline_extraction_mask,dtype=int) # casting necessary for take_along_axis forward-compatibility
    CHORD_beam,Galt_beam=leading_order_beams
    interface=BeamInterface(Galt_beam, beam_type="efield")
    az_vec=np.linspace(0,np.pi/2, Nside) # rad
    za_vec=np.linspace(0,2*np.pi, Nside) # rad
    # TODO: acquiesce to the embarassing paraellizability (although, yes, I had to do it the slow way first to figure things out)
    print("manually_track_with_Galt: antpos =",antpos)
    for i in range(Ntime):
        repointed_Galt_beam=interface.compute_response(az_array=az_vec,
                                                       za_array=za_vec,
                                                       freq_array=freqs,
                                                       az_za_grid=True,
                                                       freq_interp_kind="cubic", # this *is* the default, but it's a useful reminder in case I need to turn it up later
                                                       # interpolation_function and other kwargs are useless here because I am using AnalyticBeams, not UVBeams
                                                       )
        # literal 0th-order TODO: think about why that is probably wrong because Galt is EQUATORIALLY MOUNTED
        leading_order_beams=[CHORD_beam,repointed_Galt_beam]

        vis_i=simulate_visibilities(matvis_backend=matvis_backend,
                                    antpos=antpos,
                                    # TODO: propagate the beam type indices out to this level of wrapper to make the implementation less brittle (although this is, indeed, )
                                    freqs=freqs,
                                    times=time_vec[i:i+1], # have to explicitly array-dimensionalize this so matvis doesn't freeze up
                                    telescope_loc=telescope_loc)
        vis_i=vis_i[:,0,:] # make the (Nfreq, 1, Nbaselines) slice (Nfreq, Nbaselines)-shaped
        Galt_tracked_visibilities[:,i,:]=vis_i 

    # TODO: do not assume that the baseline ordering will always follow C-like ordering (either verify that or generalize this)
    heterogeneous_baseline_extraction_mask_flattened=np.reshape(heterogeneous_baseline_extraction_mask,(Nant**2,),order="C")
    # I flattened antennaA,antennaB into baselineAB, but now I have to tile ths to match the shape of the vis mat: (Nfreq,Ntime,Nbl)
    het_bl_mask_3d = np.tile(heterogeneous_baseline_extraction_mask_flattened, 
                             (Nfreq, Ntime, 1))
    print("mean, std of 3d mask: ",np.mean(het_bl_mask_3d),np.std(het_bl_mask_3d))
    tracked_vis_heterogeneous_only=np.take_along_axis(Galt_tracked_visibilities,
                                                      het_bl_mask_3d,
                                                      axis=2)
    print("tracked_vis_heterogeneous_only.shape=",tracked_vis_heterogeneous_only.shape)
    print("extrema of tracked_vis_heterogeneous_only: ",np.min(tracked_vis_heterogeneous_only),np.max(tracked_vis_heterogeneous_only))

    print("manually_track_with_Galt: np.sum(Galt_tracked_visibilities), np.sum(Galt_tracked_visibilities[:,:,0])",np.sum(Galt_tracked_visibilities), np.sum(Galt_tracked_visibilities[:,:,0]))
    return Galt_tracked_visibilities,tracked_vis_heterogeneous_only

# TODO: stop doing hacky outside-the-function simulation and actually use my own helper function
# similar to simulate_sky but just the bright catalogue sources
CHIME_RA=np.asarray([elem["ra"]*(np.pi/12) for elem in CHIME_match.values()]) # hrs to rad: 360/24 * pi/180 = pi/24 * 2 = pi/12
CHIME_DEC=np.asarray([elem["dec"]*np.pi/180 for elem in CHIME_match.values()]) # deg to rad
CHIME_ALPHAS=np.asarray([elem["alpha"] for elem in CHIME_match.values()])
CHIME_S600=np.asarray([elem["S600"] for elem in CHIME_match.values()])
CHIME_FLUX_ALLFREQ=((FREQS[:, np.newaxis] / FREQS[0]) ** CHIME_ALPHAS.T * CHIME_S600.T).T

# something I should've tested during the week of Sept 21st but apparently I forgor
res= binned_statistic_2d(CHIME_RA, CHIME_DEC, CHIME_S600, # bins=Nside**2, 
                         bins=[500,500],
                         range=[[0,2*np.pi], [-np.pi,np.pi]]) # dec and RA both in rad
histogram_of_simulated_point_sources=res.statistic
histogram_of_simulated_point_sources[np.isnan(histogram_of_simulated_point_sources)]=0
print("extrema of histogram: ",np.min(histogram_of_simulated_point_sources),np.max(histogram_of_simulated_point_sources))

# plot the sky map
plt.figure()
# plt.subplot(projection="mollweide")
plt.imshow(histogram_of_simulated_point_sources.T, 
           norm="symlog",
           extent=[0,2*np.pi,-np.pi,np.pi],
           origin="lower",
           cmap=cmr.torch)
plt.axhline(19*np.pi/180,c="w")
plt.axhline(79*np.pi/180,c="w")
cbar=plt.colorbar()
cbar.set_label("brightness (Jy)")
plt.xlabel("RA (rad)")
plt.ylabel("dec (rad)")
# plt.xlim(-90,90)
# plt.ylim(0,24)
plt.title("simulated sky")
plt.savefig("simulated_sky.png",dpi=500)
plt.close()

# CHORD layout
CHORD_NS_bl=8.5*u.m
CHORD_EW_bl=6.3*u.m
CHORD_N_NS=24
CHORD_N_EW=22
half_NS=CHORD_N_NS//2
half_EW=CHORD_N_EW//2

CHORD_NS_vec=CHORD_NS_bl*CHORD_N_NS*fftshift(fftfreq(CHORD_N_NS))
CHORD_EW_vec=CHORD_EW_bl*CHORD_N_EW*fftshift(fftfreq(CHORD_N_EW))
CHORD_EW_coords,CHORD_NS_coords=np.meshgrid(CHORD_EW_vec,CHORD_NS_vec)
Galt_EW=CHORD_EW_coords[half_EW,half_NS]+116*u.m # prints out an astropy value
Galt_NS=CHORD_NS_coords[half_EW,half_NS]+  6*u.m
CHORD_ant_mask=np.ones((24,22),dtype="bool")
CHORD_ant_mask[ -  1 ,   0  ]=False # NW corner observatory access road gap
CHORD_ant_mask[    6 ,  10  ]=False # N receiver hut
CHORD_ant_mask[   17 ,  10  ]=False # S receiver hut
CHORD_ant_mask[ :  6 , - 2: ]=False # SE corner too steep for antennas
CHORD_ant_mask[    6 , - 1  ]=False # extra antenna missing from SE corner

CHORD_EW=CHORD_EW_coords[CHORD_ant_mask]
CHORD_NS=CHORD_NS_coords[CHORD_ant_mask]

orientation=-1.75*np.pi/180
rot_mat= np.asarray([[np.cos(orientation),-np.sin(orientation)], 
                     [np.sin(orientation), np.cos(orientation)]])

index_of_single_CHORD_ant=0
single_CHORD_EW=CHORD_EW[index_of_single_CHORD_ant]
single_CHORD_NS=CHORD_NS[index_of_single_CHORD_ant]
holog_EW_unitless_1bl=[single_CHORD_EW.value, Galt_EW.value]
holog_NS_unitless_1bl=[single_CHORD_NS.value, Galt_NS.value]
EN_unitless_1bl = np.vstack((holog_EW_unitless_1bl, holog_NS_unitless_1bl)).T
EN_unitless_1bl = np.dot(EN_unitless_1bl, rot_mat.T)
E_unitless_1bl,N_unitless_1bl=EN_unitless_1bl.T

CHORD_EW=list(CHORD_EW)
CHORD_NS=list(CHORD_NS)
holog_EW=CHORD_EW +[Galt_EW] # should be a list by now
holog_NS=CHORD_NS +[Galt_NS]

holog_EW_unitless=[ew.value for ew in holog_EW]
holog_NS_unitless=[ns.value for ns in holog_NS]
EN_unitless = np.vstack((holog_EW_unitless, holog_NS_unitless)).T

EN_unitless = np.dot(EN_unitless, rot_mat.T)
E_unitless,N_unitless=EN_unitless.T

holog_EW=list(E_unitless*u.m) # re-formed, now rotated
holog_NS=list(N_unitless*u.m)

plt.figure()
plt.scatter(E_unitless,N_unitless)
plt.title("holog coordinates")
plt.savefig("holog_coords.png")
plt.close()

# experiments from the week of Sept 28th
if recalc:
    t1=time.time()
    # CHORD_Galt_tracking_vis,heterogeneous_only=manually_track_with_Galt(antpos=coord_arrays_to_HERA_format(E_unitless,N_unitless)) # nonsense where I was trying to simulate full CHORD at full time and frequency resolution
    CHORD_Galt_tracking_vis,heterogeneous_only=manually_track_with_Galt(antpos=coord_arrays_to_HERA_format(E_unitless_1bl,N_unitless_1bl)) # heterogeneous_only is redundant when there's only a single simulated baseline
    t2=time.time()
    print("simulating hybrid driftscan-tracking CHORD x Galt visibilities took {} s".format(t2-t1))
    print("np.sum(CHORD_Galt_tracking_vis) =",np.sum(CHORD_Galt_tracking_vis))
    np.save("CHORD_x_Galt_tracking.npy",CHORD_Galt_tracking_vis)
    # np.savez("CHORD_x_Galt_tracking_heterogeneous_only.npz", heterogeneous_only)

else:
    CHORD_Galt_tracking_vis=np.load("CHORD_x_Galt_tracking.npy")
    print("np.sum(CHORD_Galt_tracking_vis) =",np.sum(CHORD_Galt_tracking_vis))
    assert np.sum(CHORD_Galt_tracking_vis)>0, "why is this ok before saving and bad after???"
    heterogeneous_only= np.zeros_like(CHORD_Galt_tracking_vis) # np.load("CHORD_x_Galt_tracking_heterogeneous_only.npz")["arr_0"]

print("CHORD_Galt_tracking_vis.shape=",CHORD_Galt_tracking_vis.shape)
visibility_versions=[CHORD_Galt_tracking_vis,heterogeneous_only]
visibility_names=["all baselines", "heterogeneous baselines only"]
shortname=["all","het"]

# Nfreq, Ntime, Nbl

case_use=0 # this has to be 0 when i only simulate one baseline
vis=visibility_versions[case_use]
hr_after_start_of_acq=(time_vec[-1]-np.floor(time_vec[-1]))*24

# TODO: run matvis GPU backend on Fir OR find the backdoor to use multiple beam types with fftvis

src=CHIME_match["CygA"]
LST= np.copy(time_vec) # TODO: convert these to sidereal 
fs_ha = LST - src["ra"] # HA=LST-RA; at least these are both already in hours
fs_dec= src["dec"]
DRAOlat=DRAO.lat.value
assert np.abs(DRAOlat-49)<1, "was the DRAO lat actually stored in degrees?"
Deltax,Deltay= single_CHORD_EW.value-Galt_EW.value, single_CHORD_NS.value-Galt_NS.value
lambdas=c*u.m/u.s/FREQS
lambdas=lambdas.decompose()

# draco -> draco -> analysis -> beam.py -> CreateBeamStreamFromTelescope.process
u=np.atleast_1d(Deltax)[np.newaxis, :] / lambdas[:, np.newaxis] # this would've been good for more baselines
u=u[:, :, np.newaxis, np.newaxis]
u=u.value
v=np.atleast_1d(Deltay)[np.newaxis, :] / lambdas[:, np.newaxis]
v=v[:, :, np.newaxis, np.newaxis]
v=v.value
# u=Deltax[np.newaxis, :] / lambdas[:, np.newaxis] # this would've been good for more baselines
# u=u[:, :, np.newaxis, np.newaxis]
# v=Deltay[np.newaxis, :] / lambdas[:, np.newaxis]
# v=v[:, :, np.newaxis, np.newaxis]


# "Rotate the baseline distances by the telescope's rotation angle.            << this rotation angle is about literal offset from true NS-vs-EW baselines
# This assumes that the baseline distances used to beamform in the
# NS direction were NOT rotated, and hence the phase due to that rotation
# should be corrected by the beam.  Note that we can only partially
# correct for the rotation in this way, since we have already collapsed
# over NS baselines. However, this partial correction should be pretty good
# for small rotation angles and for sources near meridian."
# rot = np.deg2rad(1.75)                                                       << although I put a CHORD-like rotation angle in here, I'm not using this part, because I used actual antenna positions where the CHORD antenna already has array rotation baked in
# v = np.sin(rot) * u
# u = np.cos(rot) * u

# Calculate the phase
phi_Galt_CHORD = fringestop_phase_term( fs_ha, np.deg2rad(DRAOlat), fs_dec, 
                                        u, v 
                                        ).conj()
phi_CHORD_Galt = fringestop_phase_term( fs_ha, np.deg2rad(DRAOlat), fs_dec, 
                                        -u, -v 
                                        ).conj()
print("shape of fringestopping phase =",phi_Galt_CHORD.shape)
print("shape of reverse-engineered pruned fringestopping phase =",phi_CHORD_Galt.shape)

fringestopping_phase_terms=[1, 
                            phi_CHORD_Galt[:,0,0,:], 
                            phi_Galt_CHORD[:,0,0,:], 
                            1] # CHORD-CHORD, CHORD-Galt, Galt-CHORD, Galt-Galt << order intuited from looking at waterfalls
CHORD_ant_name="CHORD X01" # TODO: make this more robust so it's not hard-coded if I change which CHORD antenna I put in my visibility simulations
baseline_names=[CHORD_ant_name+"–"+CHORD_ant_name,
                CHORD_ant_name+"–Galt",
                "Galt–"+CHORD_ant_name,
                "Galt-Galt"]
human_friendly_t0=TIMES[0].strftime("%Y %b %d %H:%m")
# ok to set the norm from unfringestopped visibilities because, of course, applying a phase doesn't change the norm
cyclic_cmap="twilight_shifted"
sequential_cmap=cmr.sapphire

################################################################################################################################################
# NULL TEST WITH ABS2
sharednorm=LogNorm(vmin=(np.abs(vis)**2).min(),vmax=(np.abs(vis)**2).max())
fig,axs=plt.subplots( 2,4, layout="constrained", figsize=(12,5),
                      sharex=True, sharey=True                    
                      )
axs[1,0].set_xlabel("time after start of acq (hr)")
axs[1,0].set_ylabel("freq (MHz)")
row0_title_prefixes=["\n","                                                  raw |vis|$^2$\n","\n","\n"]
axs[1,1].set_title("                                                       fringestopped |vis|$^2$")
plt.suptitle("matvis-simulated CHORD-Galt observation of a Cyg A transit starting {}".format(human_friendly_t0))

for j in range(4):
    vis_for_wf=vis[:,:,j]
    wf_raw=np.abs(vis_for_wf)**2
    fringestopped_vis_for_wf=vis_for_wf*fringestopping_phase_terms[j] # wait highkey i need to verify if this is the right direction
    wf_fringestopped=np.abs(fringestopped_vis_for_wf)**2

    im= axs[0,j].imshow( wf_raw,
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   sharednorm,
                         origin= "lower",
                         cmap=sequential_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )
    axs[0,j].set_title(row0_title_prefixes[j]+baseline_names[j])
    im= axs[1,j].imshow( wf_fringestopped,
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   sharednorm,
                         origin= "lower",
                         cmap=sequential_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )

plt.colorbar(im,ax=axs,label="ADU$^2$",shrink=0.9)
plt.savefig("waterfalls_abs2_raw_and_fringestopped.png",dpi=300)
plt.close()

################################################################################################################################################
# SECOND VERSION WITH RE AND IM PARTS
Repct5=np.percentile(vis.real,5)
lt_Re = Repct5 if Repct5>0 else 10
Re_norm=SymLogNorm(lt_Re,
                   vmin=vis.real.min(),vmax=vis.real.max())
Impct5=np.percentile(vis.imag,5)
lt_Im = Impct5 if Impct5>0 else 10
Im_norm=SymLogNorm(lt_Im,
                   vmin=vis.imag.min(),vmax=vis.imag.max())
fig,axs=plt.subplots( 4,4, layout="constrained", figsize=(12,12),
                      sharex=True, sharey=True                    
                      )
axs[3,0].set_xlabel("time after start of acq (hr)")
axs[3,0].set_ylabel("freq (MHz)")
row0_title_prefixes=["\n","                                                  Re(vis)\n","\n","\n"]
axs[1,1].set_title("                                                          Im(vis)")
row2_title_prefixes=["\n","                                                fringestopped Re(vis)\n","\n","\n"]
axs[3,1].set_title("                                                       fringestopped Im(vis)")
plt.suptitle("matvis-simulated CHORD-Galt observation of a Cyg A transit starting {}".format(human_friendly_t0))

for j in range(4):
    vis_for_wf=vis[:,:,j]
    fringestopped_vis_for_wf=vis_for_wf*fringestopping_phase_terms[j] # wait highkey i need to verify if this is 

    im= axs[0,j].imshow( vis_for_wf.real,
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   Re_norm,
                         origin= "lower",
                         cmap=sequential_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )
    axs[0,j].set_title(row0_title_prefixes[j]+baseline_names[j])
    if j==3:
        plt.colorbar(im,ax=axs[0,3],label="ADU",shrink=0.9)
    im= axs[1,j].imshow( vis_for_wf.imag,
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   Im_norm,
                         origin= "lower",
                         cmap=sequential_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )
    if j==3:
        plt.colorbar(im,ax=axs[1,3],label="ADU",shrink=0.9)
    
    im= axs[2,j].imshow( fringestopped_vis_for_wf.real,
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   Re_norm,
                         origin= "lower",
                         cmap=sequential_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )
    axs[2,j].set_title(row2_title_prefixes[j])
    im= axs[3,j].imshow( fringestopped_vis_for_wf.imag,
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   Im_norm,
                         origin= "lower",
                         cmap=sequential_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )

plt.savefig("waterfalls_Re_Im_raw_and_fringestopped.png",dpi=300)
plt.close()

################################################################################################################################################
amp_norm=LogNorm(vmin=np.abs(vis).min(),vmax=np.abs(vis).max())
ph_norm=CenteredNorm(vcenter=0,halfrange=np.pi)
# THIRD VERSION WITH AMP AND PHASE
fig,axs=plt.subplots( 4,4, layout="constrained", figsize=(12,12),
                      sharex=True, sharey=True                    
                      )
axs[3,0].set_xlabel("time after start of acq (hr)")
axs[3,0].set_ylabel("freq (MHz)")
row0_title_prefixes=["\n","                                                  abs(vis)\n","\n","\n"]
axs[1,1].set_title("                                                          ph(vis)")
row2_title_prefixes=["\n","                                                fringestopped abs(vis)\n","\n","\n"]
axs[3,1].set_title("                                                       fringestopped ph(vis)")
plt.suptitle("matvis-simulated CHORD-Galt observation of a Cyg A transit starting {}".format(human_friendly_t0))

for j in range(4):
    vis_for_wf=vis[:,:,j]
    fringestopped_vis_for_wf=vis_for_wf*fringestopping_phase_terms[j] # wait highkey i need to verify if this is 

    im= axs[0,j].imshow( np.abs(vis_for_wf),
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   amp_norm,
                         origin= "lower",
                         cmap=sequential_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )
    axs[0,j].set_title(row0_title_prefixes[j]+baseline_names[j])
    if j==3:
        plt.colorbar(im,ax=axs[0,3],label="ADU",shrink=0.9)
    im= axs[1,j].imshow( np.unwrap(np.angle(vis_for_wf),axis=-1),
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   ph_norm,
                         origin= "lower",
                         cmap=cyclic_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )
    if j==3:
        plt.colorbar(im,ax=axs[1,3],label="ADU",shrink=0.9)
    
    im= axs[2,j].imshow( np.abs(fringestopped_vis_for_wf),
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   amp_norm,
                         origin= "lower",
                         cmap=sequential_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )
    axs[2,j].set_title(row2_title_prefixes[j])
    im= axs[3,j].imshow( np.unwrap(np.angle(fringestopped_vis_for_wf),axis=-1),
                         aspect= hr_after_start_of_acq/freq_range,
                         norm=   ph_norm,
                         origin= "lower",
                         cmap=cyclic_cmap,
                         extent= [ 0,              hr_after_start_of_acq,
                                   FREQS[0].value, FREQS[-1].value        ]
                         )

plt.savefig("waterfalls_amp_ph_raw_and_fringestopped.png",dpi=300)
plt.close()