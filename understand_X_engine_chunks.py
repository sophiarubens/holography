import numpy as np
from matplotlib import pyplot as plt
import h5py
import hdf5plugin # looks like it's unused but it is actually required for some HDF5 functionality

holography_directory="/Users/sophiarubens/Downloads/research/code/holography/"
single_vis_file="vis_0004224754_20260913T_130433_035210550.h5"

with h5py.File(holography_directory+single_vis_file, "r") as f:
    print(list(f.keys()))
    print(list(f.attrs))
    visibilities=f["vis"][()]
    visweights=f["vis_weight"][()]
    bfm=f["bf_mask"]
    print("bad feed mask attributes:",bfm.attrs)

    print("--- OBJECT TYPE ---")
    print(f"Type of 'bad feed mask': {type(bfm)}")
    
    print("\n--- METADATA / ATTRIBUTES ---")
    # This loops through the container and prints actual keys and values
    if len(bfm.attrs) == 0:
        print("No attributes found.")
    for key, value in bfm.attrs.items():
        print(f"Attribute Name: '{key}' -> Value: {value}")
        
    print("\n--- CONTENTS INSIDE THIS GROUP ---")
    # If it's a group, this reveals what datasets or folders live inside it
    if isinstance(bfm, h5py.Group):
        print(f"Keys inside this group: {list(bfm.keys())}")
    else:
        print("This object is a dataset, not a group folder.")

    rfi_x=f["rfi_frame_excision_enabled"][()]


print("visibilities.shape=",visibilities.shape)
print("visweights.shape=",visweights.shape)
print("bad_feed_mask=",bfm)
print("rfi_x=",rfi_x)
