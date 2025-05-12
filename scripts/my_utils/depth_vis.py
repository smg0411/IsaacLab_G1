import numpy as np
import matplotlib.pyplot as plt

depth = np.load('./camera/distance_to_camera_183_0.npy')

print(depth.shape)
plt.imshow(depth, cmap='gray')  # 혹은 cmap='plasma', 'inferno', 'viridis' 등
plt.colorbar(label='Depth Value')
plt.title("Depth Image")
plt.show()