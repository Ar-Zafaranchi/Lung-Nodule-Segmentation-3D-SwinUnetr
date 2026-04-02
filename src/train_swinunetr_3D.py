
#!pip install einops

import glob
import numpy as np

#l = glob.glob("/home/private/3D/Empty_Patches_3D/Empty_Patches_3D/CT/*.npy")
#np.shape(l)



#!pip install --upgrade monai==1.2.0
#!pip install torch torchvision
import monai
print(monai.__version__)

import torch
from monai.networks.nets import SwinUNETR
import numpy as np
import os
from sklearn.model_selection import train_test_split
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device in use:", device, "CUDA available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0), flush=True)

mask_dir = "/gpfs/ddn/users/zafara/arman/Data/Patched_3D/3d-96/Mask/"
ct_dir = "/gpfs/ddn/users/zafara/arman/Data/Patched_3D/3d-96/Normalized_CT/"

# List and pair by ID (remove '.npy')
ct_ids = [f[:-4] for f in os.listdir(ct_dir) if f.endswith('.npy')]
mask_ids = [f[:-4] for f in os.listdir(mask_dir) if f.endswith('.npy')]
patch_ids = sorted(list(set(ct_ids) & set(mask_ids)))

ct_files = [os.path.join(ct_dir, f"{pid}.npy") for pid in patch_ids]
mask_files = [os.path.join(mask_dir, f"{pid}.npy") for pid in patch_ids]

print("Total paired patches:", len(patch_ids))

# Train/validation split


ct_train, ct_val, mask_train, mask_val = train_test_split(
    ct_files, mask_files, test_size=0.2, random_state=42
)

from torch.utils.data import Dataset

class NpyPatchDataset(Dataset):
    def __init__(self, ct_files, mask_files, transform=None):
        self.ct_files = ct_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        ct = np.load(self.ct_files[idx]).astype(np.float32)  # shape: [Y, X, Z]
        mask = np.load(self.mask_files[idx]).astype(np.float32)

        # Transpose to [Z, Y, X] for model input
        ct = ct.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        # Add channel dimension
        ct = np.expand_dims(ct, 0)
        mask = np.expand_dims(mask, 0)

        sample = {"image": ct, "label": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandBiasFieldd, RandAdjustContrastd, ToTensord,
)
print("Allocated VRAM before:", torch.cuda.memory_allocated() / 1024 ** 2, "MB")

train_transforms = Compose([
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandAffined(
        keys=["image", "label"],
        prob=0.5,
        rotate_range=(0.1, 0.1, 0.1),
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest"),
    ),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0, std=0.1),
    RandBiasFieldd(keys=["image"], prob=0.3),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    ToTensord(keys=["image", "label"]),
])


val_transforms = Compose([
    ToTensord(keys=["image", "label"]),
])

# ... load and augment batch
print("Allocated VRAM after:", torch.cuda.memory_allocated() / 1024 ** 2, "MB")

# from monai.transforms import Compose, ToTensord, RandFlipd, RandRotate90d, NormalizeIntensityd

# train_transforms = Compose([
#     ToTensord(keys=["image", "label"]),
#     NormalizeIntensityd(keys=["image"]),
#     # Random flip along Z (depth) axis

# ])
# val_transforms = Compose([
#     NormalizeIntensityd(keys=["image"]),
#     ToTensord(keys=["image", "label"]),
# ])

from torch.utils.data import DataLoader

train_dataset = NpyPatchDataset(ct_train, mask_train, transform=train_transforms)
val_dataset = NpyPatchDataset(ct_val, mask_val, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=48,         # Use 24 or 48 depending on GPU
    drop_rate=0.2,
    attn_drop_rate=0.2,
    dropout_path_rate=0.2,
    normalize=True,
).to(device)

import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.6, bce_weight=0.4):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Flatten for dice
        inputs_flat = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        # Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + 1e-5) / (inputs_flat.sum() + targets_flat.sum() + 1e-5)
        dice_loss = 1 - dice
        bce_loss = self.bce(inputs, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

# Use in your training:

from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import DiceCELoss

pretrained_path = "/gpfs/ddn/users/zafara/arman/swinunetr_ssl_pretrained.pth"
weights = torch.load(pretrained_path, map_location=device)
if 'model' in weights:
    weights = weights['model']

# Remap 'encoder.' to 'swinViT.' if needed
remapped_weights = {}
for k, v in weights.items():
    if k.startswith('encoder.'):
        new_key = k.replace('encoder.', 'swinViT.', 1)
        # Fix additional '.0.' in keys if present
        for layer_num in range(1, 5):
            new_key = new_key.replace(f'layers{layer_num}.0.0.', f'layers{layer_num}.0.')
        remapped_weights[new_key] = v
    else:
        remapped_weights[k] = v

model.load_state_dict(remapped_weights, strict=False)


# loss_function = DiceCELoss(sigmoid=True)
loss_function = DiceBCELoss(dice_weight=0.6, bce_weight=0.4)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

# --------- METRICS ---------
def get_metrics(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    intersection = (pred_bin * target_bin).sum()
    dice = (2. * intersection) / (pred_bin.sum() + target_bin.sum() + eps)
    precision = intersection / (pred_bin.sum() + eps)
    sensitivity = intersection / (target_bin.sum() + eps)
    return dice.item(), precision.item(), sensitivity.item()

num_epochs = 200
best_val_loss = float("inf")
best_epoch = 0
best_weights_path = "/gpfs/ddn/users/zafara/arman/Data/Patched_3D/96*96_HAUG_0.4-0.6_lr0.0001_bs1_swinunetr.pth"

train_losses, val_losses = [], []
train_dices, val_dices = [], []
train_precisions, val_precisions = [], []
train_sensitivities, val_sensitivities = [], []
# print("1")
for epoch in range(num_epochs):
    model.train()
    train_loss = train_dice = train_precision = train_sensitivity = 0
    n_train_batches = 0
    # print("2")
    for batch_data in train_loader:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print("3")
        outputs_sig = torch.sigmoid(outputs).detach().cpu()
        labels_cpu = labels.detach().cpu()
        dice, precision, sensitivity = get_metrics(outputs_sig, labels_cpu)
        train_dice += dice
        train_precision += precision
        train_sensitivity += sensitivity
        n_train_batches += 1
    # print("4")
    avg_train_loss = train_loss / n_train_batches
    avg_train_dice = train_dice / n_train_batches
    avg_train_precision = train_precision / n_train_batches
    avg_train_sensitivity = train_sensitivity / n_train_batches

    # Validation
    model.eval()
    val_loss = val_dice = val_precision = val_sensitivity = 0
    n_val_batches = 0

    with torch.no_grad():
        for val_data in val_loader:
            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            val_outputs = model(val_images)
            loss = loss_function(val_outputs, val_labels)
            val_loss += loss.item()
            val_outputs_sig = torch.sigmoid(val_outputs).cpu()
            val_labels_cpu = val_labels.cpu()
            dice, precision, sensitivity = get_metrics(val_outputs_sig, val_labels_cpu)
            val_dice += dice
            val_precision += precision
            val_sensitivity += sensitivity
            n_val_batches += 1

    avg_val_loss = val_loss / n_val_batches
    avg_val_dice = val_dice / n_val_batches
    avg_val_precision = val_precision / n_val_batches
    avg_val_sensitivity = val_sensitivity / n_val_batches

    # Learning rate decay
    scheduler.step(avg_val_loss)

    # Print LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{num_epochs} "
          f"| Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
          f"| Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f} "
          f"| Train Prec: {avg_train_precision:.4f} | Val Prec: {avg_val_precision:.4f} "
          f"| Train Sens: {avg_train_sensitivity:.4f} | Val Sens: {avg_val_sensitivity:.4f} "
          f"| LR: {current_lr:.2e}",
          flush=True)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_weights_path)
        print(f"Best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}", flush=True)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_dices.append(avg_train_dice)
    val_dices.append(avg_val_dice)
    train_precisions.append(avg_train_precision)
    val_precisions.append(avg_val_precision)
    train_sensitivities.append(avg_train_sensitivity)
    val_sensitivities.append(avg_val_sensitivity)


print(f"Training complete. Best model was at epoch {best_epoch} with val loss {best_val_loss:.4f}", flush=True)

import matplotlib.pyplot as plt

epochs = list(range(1, num_epochs + 1))

plt.figure(figsize=(12, 10))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid(True)

# Dice plot
plt.subplot(2, 2, 2)
plt.plot(epochs, train_dices, label='Train Dice')
plt.plot(epochs, val_dices, label='Val Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.title('Dice per Epoch')
plt.legend()
plt.grid(True)

# Precision plot
plt.subplot(2, 2, 3)
plt.plot(epochs, train_precisions, label='Train Precision')
plt.plot(epochs, val_precisions, label='Val Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision per Epoch')
plt.legend()
plt.grid(True)

# Sensitivity plot
plt.subplot(2, 2, 4)
plt.plot(epochs, train_sensitivities, label='Train Sensitivity')
plt.plot(epochs, val_sensitivities, label='Val Sensitivity')
plt.xlabel('Epoch')
plt.ylabel('Sensitivity')
plt.title('Sensitivity per Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("/gpfs/ddn/users/zafara/arman/Data/Patched_3D/96_training_curves_HAUG.png")
plt.close()

'''
c = np.load("/home/private/3D/3D_patch_96/3D_Bbox_96_96/Normalized_CT/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_nodule0.npy")
m = np.load("/home/private/3D/3D_patch_96/3D_Bbox_96_96/Mask/1.3.6.1.4.1.14519.5.2.1.6279.6001.106719103982792863757268101375_nodule0.npy")

np.shape(c)

np.max(c)

import numpy as np
import matplotlib.pyplot as plt

arr = c # Shape should be (64, 64, 64)

print("Shape:", arr.shape)

# Test: Show middle slices for each axis
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(arr[arr.shape[0] // 2, :, :], cmap='gray')
axs[0].set_title('arr[Z_mid, :, :]  (axis 0)')
axs[1].imshow(arr[:, arr.shape[1] // 2, :], cmap='gray')
axs[1].set_title('arr[:, X_mid, :]  (axis 1)')
axs[2].imshow(arr[:, :, arr.shape[2] // 2], cmap='gray')
axs[2].set_title('arr[:, :, Y_mid]  (axis 2)')
plt.tight_layout()
plt.show()

plt.imshow(c[:, 10, :])

for i in range(np.shape(c)[0]):
    print(i)
    plt.figure(figsize=[20, 20])
    plt.subplot(121)
    plt.imshow(c[i, :, :], cmap='gray')
    plt.imshow(m[i, :, :], alpha=0.6, cmap='jet')
    plt.subplot(122)
    plt.imshow(c[i, :, :], cmap='gray')

    plt.show()

def windower(data, wmin=-1024, wmax=100):

    dump = data.copy()

    # Clip extreme values to within [wmin, wmax]
    dump[dump > wmax] = wmax
    dump[dump < wmin] = wmin

    # Normalize to [0,1] ensuring full scaling
    normalized = (dump - wmin) / (wmax - wmin)

    return normalized

ct_list = glob.glob("/home/private/3D/3D_patch_96/3D_Bbox_96_96/CT/*.npy")

len(ct_list)

ct_list[0]

dest = "/home/private/3D/3D_patch_96/3D_Bbox_96_96/Normalized_CT"
for i in range(len(ct_list)):


    temp = np.load(ct_list[i])
    Patient_ID2 = ct_list[i].split("/")[-1][:-4]
    temp = windower(temp)

    print(np.max(temp))
    print(np.min(temp))

    np.save(os.path.join(dest, Patient_ID2 + ".npy"), temp)
    
'''
