import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm,LogNorm
from scipy.fft import fftshift,ifftshift, fftn
import h5py,hdf5plugin # plugin required for successful reading of kotekan files
import cmasher

def get_battleship_string_from_array_indices(grid_x_idx,grid_y_idx):
    # trying to intuit the Kotekan convention...
    row_letter=chr(65+grid_y_idx)
    column_number=str(grid_x_idx)
    battleship_string=row_letter+column_number
    return battleship_string

holography_directory="/Users/sophiarubens/Downloads/research/code/holography/"
vis_idx="0004224754"
vis_timestamp="20260913T_130433_035210550"
single_vis_file="vis_"+vis_idx+"_"+vis_timestamp+".h5"
cmap_use=cmasher.fall # cmasher.sunset
cmap_use.set_bad(color="C0")

with h5py.File(holography_directory+single_vis_file) as f:
    freq = f["index_map/freq"]["centre"]      # MHz, (Nf,)
    prod = f["index_map/prod"][:]             # (input_a, input_b)
    good = f["frames_added"][:].astype(bool)  # (freq, time) validity
    vis  = f["vis"][:]                        # (freq, prod, time) complex64
    w    = f["vis_weight"][:]
    t    = f["time_center_t_inst_ns"][:] * 1e-9   # UNIX seconds

    print("list of index maps:",list(f["index_map"].keys()))

    grid_x_idx=f["index_map/grid_x_idx"][:]
    grid_y_idx=f["index_map/grid_y_idx"][:]
    ant_pos=f["index_map/dish_positions_in_grid_coords"][:]
    vis[~good[:, None, :].repeat(vis.shape[1], 1)] = np.nan

    print("N2 layout: ",f.attrs["n2_layout"])
    print("acquisition duration: ",t[-1]-t[0])
    N_dishes=f.attrs["num_dishes"]
    print("N dishes:",N_dishes)

print("vis.shape=",vis.shape) # N_freq, N_product, N_time
###################################################################################################################################################

freq_lo=freq[0]
freq_hi=freq[-1]
time_lo=t[0]
time_hi=t[-1]
Delta_freq=freq[1]-freq[0]
freq_ext=freq_hi-freq_lo
N_prod=N_dishes*2
N_bl=N_prod*(N_prod-1)
bl_lengths=np.zeros(N_bl)
print("ant_pos.shape=",ant_pos.shape)
k=0
for i in range(N_dishes):
    for j in range(i+1, N_dishes):
        dist = np.linalg.norm(ant_pos[i]-ant_pos[j])
        bl_lengths[k]=dist
        k+=1
shortest_bl=np.nanmin(bl_lengths[bl_lengths>0]) # exclude autocorrelations?! but that seems hacky and wrong if there are literal autocorrelations in this vis matrix (which there are...)
longest_bl=np.nanmax(bl_lengths)
print("shortest, longest baseline (m):",shortest_bl,longest_bl)

fr_lo=1/longest_bl
fr_hi=1/shortest_bl
del_lo=1/freq_ext
del_hi=1/Delta_freq

# experiment 1: naïve "delay spectrum" [keep baseline, freq info]
print("\n\nstart of delay spec experiment")
# vis_for_delspec=np.nanmean(vis,axis=2)
t_idx=16
vis_for_delspec=vis[:,:,t_idx]
print("vis_for_delspec.shape=",vis_for_delspec.shape)
vis_for_delspec_nan_frac=np.sum(np.isnan(vis_for_delspec))/np.prod(vis_for_delspec.shape)
print("vis for delspec nan frac=",vis_for_delspec_nan_frac)
plt.figure(figsize=(6,4),layout="constrained")
plt.imshow(np.abs(vis_for_delspec), # plot has delays as rows (y-values) and fringe rates as columns (x-values)
           origin="lower",
           cmap=cmap_use,
           extent=[shortest_bl,longest_bl,freq_lo,freq_hi],
           aspect="auto",
           norm="symlog")
ax=plt.gca()
ax.set_box_aspect(1)
cbar=plt.colorbar()
cbar.set_label("ADU$^2$")
plt.xlabel("baseline length (m)")
plt.ylabel("freq (MHz)")
# plt.title("raw delay spectrum averaged over entire acquisition"+\
plt.title("abs(vis) for a single frame"+\
          "\n"+str(100*(1-np.round(vis_for_delspec_nan_frac,5)))+"% ok; std="+\
          str(np.round(np.nanstd(vis_for_delspec),3)))
plt.savefig("abs_vis_for_delspec.png",dpi=500)
plt.close()

vis_for_delspec_nanmean=np.nanmean(vis_for_delspec)
print("nanmean of vis_for_delspec=",vis_for_delspec_nanmean)
# vis_for_delspec[np.isnan(vis_for_delspec)]=vis_for_delspec_nanmean
vis_for_delspec[np.isnan(vis_for_delspec)]=0
vis_for_delspec_tilde=fftshift(fftn(ifftshift(vis_for_delspec))) # not bothering with the volume element since it would be more trouble than it's worth to extract
                                                                 # the bin widths along each axis when I'm forming a spectrum that isn't in physical units anyway
delspec=np.abs(vis_for_delspec_tilde)**2
del_spec_nan_frac=np.sum(~np.isnan(delspec))/np.prod(delspec.shape)
print("delay spec ok fraction",del_spec_nan_frac)
rounded_str_nan_frac=str(100*np.round(del_spec_nan_frac,5))

lo=np.nanpercentile(delspec,1)
hi=np.nanpercentile(delspec,98)

plt.figure(figsize=(6,4),layout="constrained")
plt.imshow(delspec, # plot has delays as rows (y-values) and fringe rates as columns (x-values)
           origin="lower",
           cmap=cmap_use,
           extent=[-fr_hi,fr_hi,-del_hi,del_hi],
           aspect="auto",
           norm=LogNorm(vmin=1e3))
ax=plt.gca()
ax.set_box_aspect(1)
cbar=plt.colorbar()
cbar.set_label("ADU$^2$")
plt.xlabel("fringe rate (Hz)")
plt.ylabel("delay ($\mu$s)")
# plt.title("raw delay spectrum averaged over entire acquisition"+\
plt.title("raw delay spectrum for a single frame"+\
        #   "\nvisibility nans overwritten with the nanmean"
          "\nvisibility nans overwritten with zeros"
          "\n"+rounded_str_nan_frac+"% ok; std="+\
          str(np.round(np.nanstd(delspec),3))) # based on print statement bracketing, this line led to the dof warning as of 08:51 18/09/26
plt.savefig("delay_spectrum.png",dpi=500)
plt.close()
###################################################################################################################################################
# experiment 2: waterfall for a single baseline [keep freq, time info]
print("\n\nstart of waterfall experiment")
prod_idx_use=532 # 13 # 532
a_idx,b_idx=prod[prod_idx_use]
grid_x_idx_a=grid_x_idx[a_idx]
grid_x_idx_b=grid_x_idx[b_idx]
grid_y_idx_a=grid_y_idx[a_idx]
grid_y_idx_b=grid_y_idx[b_idx]
battleship_a=get_battleship_string_from_array_indices(grid_x_idx_a,grid_y_idx_a)
battleship_b=get_battleship_string_from_array_indices(grid_x_idx_b,grid_y_idx_b)
vis_for_waterfall=vis[:,prod_idx_use,:] # single correlation product (single pol of one baseline), with shape (N_freq, N_time)
vis2_for_waterfall=np.abs(vis_for_waterfall)**2
wf_nan_frac=np.sum(~np.isnan(vis2_for_waterfall))/np.prod(vis2_for_waterfall.shape)
print("waterfall ok fraction",wf_nan_frac)
product_id=prod[prod_idx_use]
print("product_id=", product_id) # this is actually going to be super annoying to translate because it's literally just the (row,col) in the visibility matrix. 
                                 # I understand why the Kotekan output pretty much had to be this way if it was going to be convenient to encode, but I need to translate it

lo=np.nanpercentile(vis2_for_waterfall,2)
mid=np.nanpercentile(vis2_for_waterfall,50)
hi=np.nanpercentile(vis2_for_waterfall,95)
print("lo, hi waterfall values:",lo,hi)

vis2_for_waterfall[~good]=np.nan

plt.figure(figsize=(6,4),layout="constrained")
plt.imshow(vis2_for_waterfall.T,
           origin="lower",
           cmap=cmap_use,
           extent=[freq_lo,freq_hi,time_lo,time_hi],
           aspect="auto",
           norm=SymLogNorm(10,vmin=lo,vmax=hi)) # plotting everything
        #    norm=SymLogNorm(10,vmin=lo,vmax=mid)) # artificially lowering the ceiling so the RFI saturates and maybe I can see some other stuff
ax=plt.gca()
ax.set_box_aspect(1)
cbar=plt.colorbar()
cbar.set_label("ADU$^2$")
plt.xlabel("freq (MHz)")
plt.ylabel("time (s)")
plt.title("waterfall for baseline "+battleship_a+"-"+battleship_b+"\n"+\
          str(100*np.round(wf_nan_frac,5))+"% ok; std="+\
          str(np.round(np.nanstd(vis2_for_waterfall),3)))
plt.savefig("waterfall_"+battleship_a+"_"+battleship_b+".png")
plt.close()
###################################################################################################################################################

# experiment 3: naïve "fringe-rate spectrum" [keep time, baseline info]
print("\n\nstart of fringe rate spec experiment")
freq_idx_use=5555 # 150
corresp_freq=freq[freq_idx_use]
string_freq=str(np.round(corresp_freq,2))
print("corresp_freq=",corresp_freq)

vis_for_frspec=vis[freq_idx_use,:,:]
vis_for_frspec_tilde=fftshift(fftn(ifftshift(vis_for_frspec)))
fringe_rate_spec=np.abs(vis_for_frspec_tilde)**2
fr_spec_nan_frac=np.sum(~np.isnan(fringe_rate_spec))/np.prod(fringe_rate_spec.shape)
print("fringe rate spec ok fraction",fr_spec_nan_frac)

plt.figure(figsize=(6,4),layout="constrained")
plt.imshow(fringe_rate_spec,
           origin="lower",
           cmap=cmap_use,
           extent=[-fr_hi,fr_hi,freq_lo,freq_hi],
           aspect="auto",
           norm="symlog")
ax=plt.gca()
ax.set_box_aspect(1)
cbar=plt.colorbar()
cbar.set_label("ADU$^2$")
plt.xlabel("fringe rate (Hz)")
plt.ylabel("frequency (MHz)")
plt.title("fringe rate spec for "+string_freq+" MHz\n"+\
          str(100*np.round(wf_nan_frac,5))+"% ok; std="+\
          str(np.round(np.nanstd(fringe_rate_spec),3)))
plt.savefig("fringe_rate_spectrum_{}_MHz.png".format(string_freq))
plt.close()