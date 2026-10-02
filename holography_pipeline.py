import numpy as np
from matplotlib import pyplot as plt

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity
from astropy import units as u

import matvis

import healpy as hp
from pyuvdata.analytic_beam import AiryBeam
from pyuvdata import BeamInterface
from scipy.fft import fftfreq,fftshift
from scipy.stats import binned_statistic_2d
import time
import cmasher

# TODO: generalize to more realistic values, chunk up the band with different Nsides, etc. 
NFREQS=100
GLOBAL_NSIDE=512
FREQS=np.linspace(300,600,NFREQS)*u.MHz
NTIMES=100
JD0=2461274
JD1=JD0+0.05
time_vec=np.linspace(JD0, JD1, NTIMES)
TIMES=Time(time_vec, format="jd", scale="utc") 
DRAO=EarthLocation.of_site("drao") # coordinates in the astropy database are for Galt, but this is convenient and good enough for now
print(DRAO)
omega_e=np.pi/12 # 2pi rad / 24 hr = pi/12

# polarized visibility matrices are indexed as (N freqs, N times, N feed i, N feed j, N baselines)
# unpolarized visibility matrices are indexed as (N freqs, N times, N antennas, N antennas) 
#    AnalyticBeam objects don't support polarization
#    in practice, the returned visibilities are actually (N freqs, N times, N baselines)

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
    # TODO: make this safe for cases where Galt is not necessarily the last beam type in the list? could be, but not a priority... this is just the way I'm simulating things

    if leading_order_beams is None:
        # TODO: add a new keyword arg so I can adjust the nominal pointing (CHORD ~Cyg A or something and Galt starting at the correct RA/dec for the repointing to be physical)
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

    return Galt_tracked_visibilities,tracked_vis_heterogeneous_only

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

print("histogram_of_simulated_point_sources.shape=",histogram_of_simulated_point_sources.shape)

# plot the sky map
plt.figure()
# plt.subplot(projection="mollweide")
plt.imshow(histogram_of_simulated_point_sources.T, 
           norm="symlog",
           extent=[0,2*np.pi,-np.pi,np.pi],
           origin="lower",
           cmap=cmasher.torch)
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
CHORD_EW=list(CHORD_EW)
CHORD_NS=list(CHORD_NS)
holog_EW=CHORD_EW +[Galt_EW] # should be a list by now
holog_NS=CHORD_NS +[Galt_NS]

holog_EW_unitless=[ew.value for ew in holog_EW]
holog_NS_unitless=[ns.value for ns in holog_NS]
EN_unitless = np.vstack((holog_EW_unitless, holog_NS_unitless)).T

orientation=-1.75*np.pi/180
rot_mat= np.asarray([[np.cos(orientation),-np.sin(orientation)], 
                     [np.sin(orientation), np.cos(orientation)]])
EN_unitless = np.dot(EN_unitless, rot_mat.T)
E_unitless,N_unitless=EN_unitless.T

holog_EW=list(E_unitless*u.m) # re-formed, now rotated
holog_NS=list(N_unitless*u.m)

plt.figure()
plt.scatter(E_unitless,N_unitless)
plt.title("holog coordinates")
plt.savefig("holog_coords.png")
plt.close()


# experiments from the week of Sept 21st
# t1=time.time()
# vis_matvis_cpu=simulate_visibilities(simulator="matvis",
#                                      antpos=coord_arrays_to_HERA_format(E_unitless,N_unitless),
#                                      matvis_backend="cpu")
# t2=time.time()
# print("fftvis CPU simulation took {} s".format(t2-t1))
# np.savez("matvis_cpu_holog.npz",vis_matvis_cpu)

# experiments from the week of Sept 28th
t1=time.time()
CHORD_Galt_tracking_vis,heterogeneous_only=manually_track_with_Galt(antpos=coord_arrays_to_HERA_format(E_unitless,N_unitless))
t2=time.time()
print("simulating hybrid driftscan-tracking CHORD x Galt visibilities took {} s".format(t2-t1))
np.savez("CHORD_x_Galt_tracking.npz",CHORD_Galt_tracking_vis)
np.savez("CHORD_x_Galt_tracking_heterogeneous_only.npz") # TODO: figure out why this file in particular is empty when I import it from another script even though the shape and absence of nans look good here

visibility_versions=[CHORD_Galt_tracking_vis,heterogeneous_only]
visibility_names=["all baselines", "heterogeneous baselines only"]
shortname=["all","het"]

# Nfreq, Ntime, Nbl

for i,vis in enumerate(visibility_versions):
    # waterfall: keep time and frequency
    baseline_ab=6223
    vis_for_wf=vis[:,:,baseline_ab]
    wf=np.abs(vis_for_wf)**2
    wfs0,wfs1=wf.shape

    plt.figure(layout="constrained",figsize=(6,4))
    plt.imshow(wf,aspect=wfs1/wfs0,norm="log")
    plt.colorbar()
    plt.title("tracking waterfall\n"+visibility_names[i])
    plt.xlabel("time (s)")
    plt.ylabel("freq (MHz)")
    print("CHECK TRANSPOSITION OF WATERFALL")
    plt.savefig("waterfall_{}_{}.png".format(baseline_ab,shortname[i]))
    plt.close()

# TODO: run matvis GPU backend on Fir OR find the backdoor to use multiple beam types with fftvis