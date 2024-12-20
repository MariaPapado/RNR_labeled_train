import numpy as np
from PIL import Image
import glob

splits = ['train', 'test', 'val']
splits = ['train', 'val']
# Load images as numpy arrays
images = []
for split in splits:
    image_paths = glob.glob("/home/mariapap/DATASETS/dataset_FP_v1/PATCHES/{}/im1/*.png".format(split))  # Adjust the path and extension as necessary

    # Initialize a list to store all image arrays

    for path in image_paths:
        image = Image.open(path)
        image_np = np.array(image)
        images.append(image_np)

    image_paths = glob.glob("/home/mariapap/DATASETS/dataset_FP_v1/PATCHES/{}/im2/*.png".format(split))  # Adjust the path and extension as necessary

    # Initialize a list to store all image arrays

    for path in image_paths:
        image = Image.open(path)
        image_np = np.array(image)
        images.append(image_np)

# Stack all images into one numpy array
# Assuming all images are of the same size and in RGB format
image_stack = np.stack(images, axis=0)  # Shape: (num_images, height, width, 3)
image_stack = image_stack[:,32:96,32:96,:]


print('aaaa', image_stack.shape)
# Compute the mean and std for each channel across all images
mean = np.mean(image_stack, axis=(0, 1, 2))
std = np.std(image_stack, axis=(0, 1, 2))

print(f"Mean (R, G, B): {mean}")
print(f"Standard Deviation (R, G, B): {std}")

print(np.mean(image_stack[:,:,:,0]))
