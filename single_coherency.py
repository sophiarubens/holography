import numpy as np
from matplotlib import pyplot as plt
from _vis_render import edges,j2000_ns_to_datetime
# from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import cmasher as cm
from matplotlib.colors import CenteredNorm


plt.rcParams.update({'font.size': 16})


t_ns=np.genfromtxt("t_ns.txt")
t_hr = (t_ns - t_ns[0]) * 1e-9 / 3600.0
freq_lo=np.genfromtxt("freq_lo.txt")

te_full, fe = edges(t_hr), edges(freq_lo)
print("te_full.shape=",te_full.shape)
LOCAL_TZ=LOCAL_TZ = ZoneInfo("America/Vancouver")
t0_local = j2000_ns_to_datetime(t_ns[:1], LOCAL_TZ)[0]

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
te=te_full[tlo:thi]
print("te.shape=",te.shape)
coh=Vab.real/np.sqrt(Vaa*Vbb)
assert np.nansum(coh.imag)<1e-9
coh=coh.real
delta_te=te[-1]-te[0]
delta_fe=fe[-1]-fe[0]
off=2
p_hi=np.nanpercentile(coh,100-off)
p_lo=np.nanpercentile(coh,off)
hr=0.5*(p_hi-p_lo)
print("p_lo, p_hi, hr =",p_lo, p_hi, hr)
t0_local_short=t0_local.replace(microsecond=0)
t0_local_short=t0_local_short.replace(tzinfo=None)

plt.figure(layout="constrained",figsize=(5,4))
plt.imshow(coh,
           cmap="twilight_shifted",
        #    cmap=cm.seasons,
           norm=CenteredNorm(halfrange=hr),
           origin="lower",
           extent=[te[0], te[-1], fe[0], fe[-1]],
           aspect=delta_te/delta_fe
           )
plt.colorbar(shrink=0.89)
plt.xlabel("time after start of acq (hr)")
plt.ylabel("freq (MHz)")
# plt.title("Coherency of baseline {}-{}\nAcquisition {}\nPointing dec=40º.7 (CygA)".format(a,b,t0_local_short)) # pointing hard-coded because this index map isn't implemented yet in the pf kotekan format
plt.savefig("single_coherency_{}_{}.png".format(a,b),dpi=250)