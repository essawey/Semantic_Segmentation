import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image
import os
import cv2
import math

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
    image = np.transpose(image, (1, 2, 0)) # (h, w, c)
    plt.figure(figsize=(6,6))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
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

    labels_idx = {label_idx: label for label_idx, label in enumerate(masksLabels)}

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    channel_images = []

    for channel_index in range(mask.shape[0]):  # Now channels are the first dimension (c)
        max_value = np.max(mask[channel_index, :, :])  # Accessing the mask for each channel
        colors = plt.cm.get_cmap('tab20', int(max_value + 1))

        row = channel_index // 3
        col = channel_index % 3

        ax = axes[row, col]
        im = ax.imshow(mask[channel_index, :, :], cmap=colors, vmin=0, vmax=max_value)
        ax.set_title(f'Channel {channel_index} : {list(labels_idx.values())[channel_index]}')
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
