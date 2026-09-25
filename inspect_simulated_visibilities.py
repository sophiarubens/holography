import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fftshift,ifftshift,fftn

def collapse_antenna_axes_to_baseline_axis(visibilities):
    assert visibilities.ndim==4 # TODO: make this function less specific to unpolarized visibility matrices
    Nfreq,Ntime,N_ant,_=visibilities.shape[-1]
    vismat=np.zeros((Nfreq,Ntime,N_ant*(N_ant-1)))
    k=0
    for i in range(N_ant):
        for j in range(N_ant):
            vismat[:,:,k]=visibilities[:,:,i,j]
    return vismat

npzfiles=np.load("matvis_cpu_holog.npz")
print("npzfiles.files =",npzfiles.files)
vis=npzfiles["arr_0"]
print("vis.shape=",vis.shape)
# Nfreq, Ntime, Nant, Nant

# waterfall: keep time and frequency
baseline_ab=6223
vis_for_wf=vis[:,:,baseline_ab]
wf=np.abs(vis_for_wf)**2
wfs0,wfs1=wf.shape

plt.figure(layout="constrained",figsize=(6,4))
plt.imshow(wf,aspect=wfs1/wfs0,norm="log")
plt.colorbar()
plt.title("waterfall for simple matvis CHORD + side antenna sim")
plt.xlabel("time (s)")
plt.ylabel("freq (MHz)")
print("CHECK TRANSPOSITION OF WATERFALL")
plt.savefig("waterfall_{}.png".format(baseline_ab))
plt.close()

# delay spec: keep frequency and baseline -> pick a single timestamp and then reshape the two antenna dimensions to be a single baseline dimension
time_idx=0
vis_for_delay_spec=vis[:,time_idx,:]
delay_transformed_vis=fftshift(fftn(ifftshift(vis_for_delay_spec),norm="backward")) # ok to transform and shift over both axes
                                                                                    # not worrying abt area element because this is just a dimensionless exercise for now
delay_spec=np.abs(delay_transformed_vis)**2

print("number of nans, infs, Nones in vis_for_delay_spec =",np.sum(np.isnan(vis_for_delay_spec)),
                                                            np.sum(np.isinf(vis_for_delay_spec)),
                                                            np.sum(np.nonzero(vis_for_delay_spec==None)))
print("number of nans, infs, Nones in delay_spec =",np.sum(np.isnan(delay_spec)),
                                                            np.sum(np.isinf(delay_spec)),
                                                            np.sum(np.nonzero(delay_spec==None)))
print("vis_for_delay_spec.shape =",vis_for_delay_spec.shape)
dss0,dss1=vis_for_delay_spec.shape
print("mean, std of vis_for_delay_spec =",np.mean(vis_for_delay_spec),np.std(vis_for_delay_spec))
fig,axs=plt.subplots(1,2,layout="constrained",figsize=(8,4))
im=axs[0].imshow(np.abs(vis_for_delay_spec),aspect=dss1/dss0)
plt.colorbar(im,ax=axs[0])
axs[0].set_xlabel("baseline length (m)")
axs[0].set_ylabel("freq (MHz)")
axs[0].set_title("abs( frequency-baseline visibility slice )")

im=axs[1].imshow(delay_spec,aspect=dss1/dss0,norm="log")
plt.colorbar(im,ax=axs[1])
axs[1].set_xlabel("fringe rate (1/m)")
axs[1].set_ylabel("delay (μs)")
axs[1].set_title("delay spec")
axs[1].set_xlim(0,dss0)
axs[1].set_ylim(0,dss1)
print("CHECK TRANSPOSITION OF DELAY SLICES")

plt.suptitle("single time step")
plt.savefig("time_slice_{}.png".format(time_idx))
plt.close()

# fringe rate spec: keep time and baseline and then reshape the two antenna dimensions to be a single baseline dimension
freq_idx=0
vis_for_fr_spec=vis[freq_idx,:,:]
frs0,frs1=vis_for_fr_spec.shape
fr_transformed_vis=fftshift(fftn(ifftshift(vis_for_fr_spec),norm="backward"))
fr_spec=np.abs(fr_transformed_vis)**2

fig,axs=plt.subplots(1,2,layout="constrained",figsize=(8,4))
im=axs[0].imshow(np.abs(vis_for_fr_spec),aspect=frs1/frs0)
plt.colorbar(im,ax=axs[0])
axs[0].set_xlabel("time (s)")
axs[0].set_ylabel("baseline (m)")
axs[0].set_title("abs( time-baseline visibility slice )")

im=axs[1].imshow(fr_spec,aspect=frs1/frs0,norm="log")
plt.colorbar(im,ax=axs[1])
axs[1].set_xlabel("freq (MHz)")
axs[1].set_ylabel("fringe rate (1/m)")
axs[1].set_title("fringe-rate spec")
print("CHECK TRANSPOSITION OF FR SLICES")

plt.suptitle("single freq channel")
plt.savefig("freq_slice_{}.png".format(freq_idx))
plt.close()