import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image
import os
import cv2
import math
import torch
import numpy as np

def binarizeChannel(masks):

    masksNum, channels, _, _ = masks.shape

    masks_binary = np.empty_like(masks)

    # Loop through each mask and channel
    for mask_idx in range(masksNum):
        for channel_idx in range(channels):

            current_channel = masks[mask_idx, channel_idx]

            # set higher values than 1 to 1
            current_channel[current_channel > 1] = 1

            masks_binary[mask_idx, channel_idx] = current_channel

    return masks_binary

def show_image(image): 

    means = [0.7039875984191895, 0.5724194049835205, 0.7407296895980835]
    stds = [0.12305392324924469, 0.16210812330245972, 0.14659656584262848]
    
    means_tensor = torch.tensor(means).view(3, 1, 1)  # Shape: [3, 1, 1]
    stds_tensor = torch.tensor(stds).view(3, 1, 1)    # Shape: [3, 1, 1]

    # Reverse normalization: (image * std) + mean
    image = image * stds_tensor + means_tensor
    
    image = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(6,6))
    plt.axis('off')

    plt.imshow(image)
    plt.show()
    

def show_mask(mask):
    masksLabels = [
        'Neoplastic cells',
        'Inflammatory',
        'Connective/Soft tissue cells',
        'Dead Cells',
        'Epithelial',
        'Background',
    ]

    # Ensure mask is a numpy array
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    # Ensure mask shape is (6, height, width)
    if mask.shape[0] != 6:
        if 1 in mask.shape:
            mask = np.squeeze(mask)
        mask = np.eye(6)[mask]  # One-hot encode to (height, width, 6)
        mask = np.moveaxis(mask, -1, 0)  # Rearrange to (6, height, width)

    labels_idx = {label_idx: label for label_idx, label in enumerate(masksLabels)}

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    channel_images = []

    for channel_index in range(mask.shape[0]):
        max_value = np.max(mask[channel_index, :, :])
        colors = plt.cm.get_cmap('tab20', int(max_value + 1))

        row = channel_index // 3
        col = channel_index % 3

        ax = axes[row, col]
        im = ax.imshow(mask[channel_index, :, :], cmap=colors, vmin=0, vmax=max_value)
        ax.set_title(f'Channel {channel_index} : {labels_idx[channel_index]}')
        ax.axis('off')
        channel_images.append(im)

    cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.03])
    fig.colorbar(channel_images[0], cax=cbar_ax, orientation='horizontal')
    plt.tight_layout()
    plt.show()


def create_patches(image_dir, patch_size, target_dir):
    for path, _, _ in sorted(os.walk(image_dir)):
        relative_path = os.path.relpath(path, image_dir)
        target_path = Path(target_dir) / relative_path
        target_path.mkdir(parents=True, exist_ok=True)

        images_index, masks_index = 0, 0

        images = sorted(os.listdir(path))
        for image_name in images:
            if image_name.endswith(".png") or image_name.endswith(".npy"):
                if image_name.endswith(".npy"):
                    image = np.load(os.path.join(path, image_name))
                    image = np.transpose(image, (1, 2, 0)) # (h, w, c)
                    image = np.argmax(image, axis=-1) # (h, w)
                    image = image.astype(np.uint8)
                else:
                    image = cv2.imread(os.path.join(path, image_name))
                size_X, size_Y = math.ceil(image.shape[1]/patch_size), math.ceil(image.shape[0]/patch_size)
                pad_X, pad_Y = (patch_size * size_X - image.shape[1]) / (size_X - 1), (patch_size * size_Y - image.shape[0]) / (size_Y - 1)
                image = Image.fromarray(image)
                top = 0
                for y in range(size_Y):
                    left = 0
                    for x in range(size_X):
                        crop_image = transforms.functional.crop(image, top, left, patch_size, patch_size)
                        crop_image = np.array(crop_image)
                        if image_name.endswith('.png'):
                            patch_name = f"{Path(image_name).stem}_patch{images_index}.png"
                            cv2.imwrite(str(target_path / patch_name), crop_image)
                            images_index += 1
                        else:
                            patch_name = f"{Path(image_name).stem}_patch{masks_index}.npy"
                            np.save(str(target_path / patch_name), crop_image)
                            masks_index += 1
                        left = left + patch_size - pad_X
                    top = top + patch_size - pad_Y
