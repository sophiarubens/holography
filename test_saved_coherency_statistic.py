import numpy as np
from matplotlib import pyplot as plt

coh=np.load("/Users/sophiarubens/Downloads/research/code/holography/coherency_B03X_B04X.npy")
plt.figure()
plt.imshow(coh)
plt.show()