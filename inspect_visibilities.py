import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm

path="/Users/sophiarubens/Downloads/research/code/holography/"
a="B05X"
b="B08X"
Vaa_full=np.load(path+"V_"+a+"_"+a+".npy")
Vab_full=np.load(path+"V_"+a+"_"+b+".npy")
Vbb_full=np.load(path+"V_"+b+"_"+b+".npy")

tlo=800
thi=1800
Vaa=Vaa_full[:,tlo:thi]
Vab=Vab_full[:,tlo:thi]
Vbb=Vbb_full[:,tlo:thi]

quantities= [[np.abs(Vaa), "abs(Vaa)"],
             [np.abs(Vab), "abs(Vab)"], 
             [np.abs(Vbb), "abs(Vbb)"],

             [Vaa.real,    "Re(Vaa)"],
             [Vab.real,    "Re(Vab)"],
             [Vbb.real,    "Re(Vbb)"],

             [Vaa.imag,    "Im(Vaa)"],
             [Vab.imag,    "Im(Vab)"],
             [Vbb.imag,    "Im(Vbb)"],

             [Vab.real/np.sqrt(Vaa*Vbb), "Re(Vab)/Re(sqrt(Vaa*Vbb))"],
             [Vaa.real/np.sqrt(Vaa*Vbb), "Re(Vaa)/Re(sqrt(Vaa*Vbb))"],
             [Vbb.real/np.sqrt(Vaa*Vbb), "Re(Vbb)/Re(sqrt(Vaa*Vbb))"],

            #  [Vaa.imag-Vab.imag,         "Im(Vaa)-Im(Vab)"],
            #  [Vaa.imag-Vbb.imag,         "Im(Vaa)-Im(Vbb)"],
            #  [Vab.imag-Vbb.imag,         "Im(Vab)-Im(Vbb)"],

            #  [Vaa.real-Vab.real,         "Re(Vaa)-Re(Vab)"],
            #  [Vaa.real-Vbb.real,         "Re(Vaa)-Re(Vbb)"],
            #  [Vab.real-Vbb.real,         "Re(Vab)-Re(Vbb)"],

            #  [Vaa.imag/Vab.imag,         "Im(Vaa)/Im(Vab)"],
            #  [Vaa.imag/Vbb.imag,         "Im(Vaa)/Im(Vbb)"],
            #  [Vab.imag/Vbb.imag,         "Im(Vab)/Im(Vbb)"],
             
            #  [Vaa.real/Vab.real,         "Re(Vaa)/Re(Vab)"],
            #  [Vaa.real/Vbb.real,         "Re(Vaa)/Re(Vbb)"],
            #  [Vab.real/Vbb.real,         "Re(Vab)/Re(Vbb)"],
             ]

Nv,Nh=3,len(quantities)//3
fig,axs=plt.subplots(Nv,Nh,layout="constrained",figsize=(8,16))

k=0
for j in range(Nh):
    for i in range(Nv):
        qty,qtyname=quantities[k]
        print("k=",k)
        if np.all(qty)==np.nan:
            qty=np.zeros_like(qty)
        axs[i,j].set_xlabel("UTC time (s)")
        axs[i,j].set_ylabel("freq (MHz)")

        print("qty.dtype=",qty.dtype)
        if qty.dtype==np.complex64:
            if np.nansum(qty.imag)<1e-9:
                qty=qty.real
        off=1
        phi=np.nanpercentile(qty,100-off)
        plo=np.nanpercentile(qty,off)
        mid=np.nanmean(qty)
        norm=CenteredNorm(vcenter=np.nanmean(qty),
                          halfrange=0.5*(phi-plo))
        if plo>phi or mid>plo:
            norm=None
        im=axs[i,j].imshow(qty, 
                           origin="lower",
                           cmap="twilight_shifted", norm=norm)
        plt.colorbar(im,ax=axs[i,j])
        axs[i,j].set_title(qtyname)
        k+=1
plt.suptitle("inspecting baseline {}-{}".format(a,b))
plt.savefig("inspect_baseline_{}_{}.png".format(a,b),dpi=500)