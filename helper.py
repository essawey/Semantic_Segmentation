import numpy as np
import matplotlib.pyplot as plt

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


def plot(mask):

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

    for channel_index in range(mask.shape[2]):
        max_value = np.max(mask[:, :, channel_index])
        colors = plt.cm.get_cmap('tab20', int(max_value + 1))

        row = channel_index // 3
        col = channel_index % 3

        ax = axes[row, col]
        im = ax.imshow(mask[:, :, channel_index], cmap=colors, vmin=0, vmax=max_value)
        ax.set_title(f'Channel {channel_index} : {list(labels_idx.values())[channel_index]}')
        ax.axis('off')
        channel_images.append(im)

    cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.03])
    fig.colorbar(channel_images[0], cax=cbar_ax, orientation='horizontal')
    plt.tight_layout()
    plt.show()
