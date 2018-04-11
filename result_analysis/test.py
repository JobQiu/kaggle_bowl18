import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

masks = np.load('difficultone.npy')
masks2 = np.swapaxes(masks,1,2)
masks2 = np.swapaxes(masks2,0,1)
for i in range(16):
    plt.figure()
    plt.imshow(masks2[i])


#%%
    
