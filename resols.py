import healpy as hp
import numpy as np

Nsides=np.logspace(0,15,num=16,base=2)
print("Nsides=",Nsides)
resols=hp.nside2resol(Nsides)
print("resols=",resols)