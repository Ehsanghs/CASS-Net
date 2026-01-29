import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import pydicom
from pathlib import Path
import re

class AISDataset(Dataset):
    def __init__(self, data_dir, patient_ids, transform=None, 
                 global_hu_clip=(-100, 300), 
                 windows=[(40, 80), (80, 200)]):
        """
        data_dir: Path to root dataset folder
        patient_ids: List of patient IDs to include
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.patient_ids = patient_ids
        self.transform = transform
        
        self.hu_min, self.hu_max = global_hu_clip
        self.windows = windows # [(center, width), ...]
        self.hu_divisor = 100.0
        
        self.samples = self._prepare_samples()
        print(f"Dataset initialized with {len(self.samples)} samples from {len(patient_ids)} patients.")

    def _prepare_samples(self):
        samples = []
        for pid in self.patient_ids:
            # Logic to pair slices with neighbors (Same logic as provided code)
            # Simplified here for brevity, assumes folder structure: /images/{pid}/CT/*.dcm
            # ... (Insert your data parsing logic here from the original script)
            pass 
        return samples

    def load_dicom_and_process(self, path):
        try:
            dcm = pydicom.dcmread(str(path))
            image = dcm.pixel_array.astype(np.float32)
            slope = getattr(dcm, 'RescaleSlope', 1.0)
            intercept = getattr(dcm, 'RescaleIntercept', 0.0)
            image = image * slope + intercept
            
            # Clipping and Normalization (Sec 3.2)
            image = np.clip(image, self.hu_min, self.hu_max)
            image = image / self.hu_divisor
            return image
        except:
            return None

    def apply_window(self, norm_img, center_raw, width_raw):
        # Convert raw center/width to normalized scale
        c = center_raw / self.hu_divisor
        w = width_raw / self.hu_divisor
        
        img_min = c - w / 2
        img_max = c + w / 2
        windowed = np.clip(norm_img, img_min, img_max)
        # Normalize to 0-1 range
        return (windowed - img_min) / (img_max - img_min + 1e-6)

    def __getitem__(self, idx):
        # ... (Implementation of loading center + neighbors) ...
        # Based on your script, you load center, prev, next.
        
        # Example channel construction:
        # ch1: Prev Slice (Stroke Window)
        # ch2: Curr Slice (Stroke Window)
        # ch3: Curr Slice (Brain Window)
        # ch4: Next Slice (Stroke Window)
        
        # Assuming we have loaded `img_prev`, `img_curr`, `img_next` (normalized)
        
        # Stroke Window (40, 80) -> Index 0 in self.windows
        sw_c, sw_w = self.windows[0]
        # Brain Window (80, 200) -> Index 1 in self.windows
        bw_c, bw_w = self.windows[1]
        
        c1 = self.apply_window(img_prev, sw_c, sw_w)
        c2 = self.apply_window(img_curr, sw_c, sw_w)
        c3 = self.apply_window(img_curr, bw_c, bw_w)
        c4 = self.apply_window(img_next, sw_c, sw_w)
        
        image = np.stack([c1, c2, c3, c4], axis=-1)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return {'image': image, 'mask': mask}

    def __len__(self):
        return len(self.samples)