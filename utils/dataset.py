import os
import glob
import numpy as np
import pydicom
import cv2
import torch
from torch.utils.data import Dataset

class AISDataset(Dataset):
    def __init__(self, data_dir, patient_ids, transform=None):
        """
        Args:
            data_dir (str): Root directory of the dataset.
            patient_ids (list): List of patient IDs to include in this split.
            transform (albumentations.Compose): Transformations for data augmentation/preprocessing.
        """
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.transform = transform
        
        # Hyperparameters according to the manuscript (Algorithm 1)
        self.hu_clip_range = (-100, 300)
        self.hu_divisor = 100.0
        self.stroke_window = (40, 80)  # (Center, Width)
        self.brain_window = (80, 200)  # (Center, Width)
        
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        """
        Organizes all slices per patient sequentially to allow 2.5D construction.
        Handles adjacent slice replication at volume boundaries.
        """
        samples = []
        for pid in self.patient_ids:
            # Expected structure: data_dir/images/pid/ and data_dir/masks/pid/
            patient_img_dir = os.path.join(self.data_dir, 'images', str(pid))
            patient_mask_dir = os.path.join(self.data_dir, 'masks', str(pid))
            
            if not os.path.exists(patient_img_dir):
                continue
                
            # Retrieve and sort DICOM files (assuming sequential naming or instance numbers)
            dcm_files = sorted(glob.glob(os.path.join(patient_img_dir, '*.dcm')))
            num_slices = len(dcm_files)
            
            for i in range(num_slices):
                # Replicate target slice at boundaries if adjacent slices are unavailable
                prev_idx = max(0, i - 1)
                next_idx = min(num_slices - 1, i + 1)
                
                base_name = os.path.splitext(os.path.basename(dcm_files[i]))[0]
                mask_path = os.path.join(patient_mask_dir, f"{base_name}.png") 
                
                samples.append({
                    'center_path': dcm_files[i],
                    'prev_path': dcm_files[prev_idx],
                    'next_path': dcm_files[next_idx],
                    'mask_path': mask_path
                })
        return samples

    def _read_and_normalize_dicom(self, path):
        """Reads a DICOM file, applies RescaleSlope/Intercept, clips, and normalizes."""
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)
        
        slope = getattr(dcm, 'RescaleSlope', 1.0)
        intercept = getattr(dcm, 'RescaleIntercept', 0.0)
        image = image * slope + intercept
        
        # Step 1: Clip intensities to -100 to 300 HU and normalize by 100
        image = np.clip(image, self.hu_clip_range[0], self.hu_clip_range[1])
        image = image / self.hu_divisor
        
        return image

    def _apply_window(self, norm_img, window):
        """Applies a specific window (center, width) to the normalized image."""
        center, width = window
        # Convert window settings to the normalized scale
        c_norm = center / self.hu_divisor
        w_norm = width / self.hu_divisor
        
        img_min = c_norm - (w_norm / 2.0)
        img_max = c_norm + (w_norm / 2.0)
        
        windowed = np.clip(norm_img, img_min, img_max)
        # Scale to [0, 1]
        windowed = (windowed - img_min) / (img_max - img_min + 1e-6)
        return windowed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load normalized HU images
        center_img = self._read_and_normalize_dicom(sample['center_path'])
        prev_img = self._read_and_normalize_dicom(sample['prev_path'])
        next_img = self._read_and_normalize_dicom(sample['next_path'])
        
        # 2. Apply windows according to manuscript
        ch1 = self._apply_window(prev_img, self.stroke_window)     # Si-1 SW
        ch2 = self._apply_window(center_img, self.stroke_window)   # Si SW
        ch3 = self._apply_window(center_img, self.brain_window)    # Si BW
        ch4 = self._apply_window(next_img, self.stroke_window)     # Si+1 SW
        
        # 3. Construct the 4-channel 2.5D input tensor (H, W, 4)
        image = np.stack([ch1, ch2, ch3, ch4], axis=-1).astype(np.float32)
        
        # 4. Load ground truth mask (or generate zero mask for lesion-negative slices)
        if os.path.exists(sample['mask_path']):
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32) # Binarize
        else:
            mask = np.zeros(center_img.shape, dtype=np.float32)
            
        # 5. Apply online augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask has channel dimension (1, H, W) for loss calculation
        if isinstance(mask, np.ndarray):
            mask = np.expand_dims(mask, axis=0)
            mask = torch.from_numpy(mask)
        elif isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return {'image': image, 'mask': mask}
