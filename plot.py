import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import numpy as np

ids = os.listdir('./R_wrong/mask/')

for id in ids:
  im1 = Image.open('./R_wrong/before/{}'.format(id))
  im2 = Image.open('./R_wrong/after/{}'.format(id))
  mask = Image.open('./R_wrong/mask/{}'.format(id))

  img1, img2, img3 = np.array(im1), np.array(im2), np.array(mask)

  # Create a figure
  plt.figure(figsize=(15, 5))

  # Plot first image
  plt.subplot(1, 3, 1)
  plt.imshow(img1)
  plt.axis('off')  # Turn off axis
  plt.title('Image 1')

  # Plot second image
  plt.subplot(1, 3, 2)
  plt.imshow(img2)
  plt.axis('off')
  plt.title('Image 2')

  # Plot third image
  plt.subplot(1, 3, 3)
  plt.imshow(img3)
  plt.axis('off')
  plt.title('Mask')

  # Display the plot
  plt.tight_layout()
  plt.show()
