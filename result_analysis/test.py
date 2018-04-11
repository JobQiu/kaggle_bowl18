import matplotlib.pyplot as plt
import numpy as np

masks = np.load('difficultone.npy')
masks2 = np.swapaxes(masks,1,2)
masks2 = np.swapaxes(masks2,0,1)
plt.imshow(masks2[0])


#%%