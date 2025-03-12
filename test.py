import numpy as np

data = np.load('/Users/dongbaiqi/Desktop/higp-main/data/fish_data.npz')

# for key in data:
#     print(np.shape(data[key]))

print(data['Fold_data'].shape)