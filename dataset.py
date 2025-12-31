import os
import torch
import glob
from torch.utils.data import Dataset
from torchvision import transforms

import torchaudio.transforms as T_audio


class GTZANDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False, time_mask = 0, freq_mask = 0, time_shift = 0):

        self.split_dir = os.path.join(root_dir, split)
        self.files = glob.glob(os.path.join(self.split_dir, "*.pt"))

        self.augment = augment
        self.time_mask = int(time_mask)
        self.freq_mask = int(freq_mask)
        self.time_shift = int(time_shift)

        # ImageNet Normalization
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


        # SPEC-AUGMENT AUGMENTATION FUNCTIONS
        self.time_masker = T_audio.TimeMasking(time_mask_param = self.time_mask) if self.time_mask > 0 else None
        self.freq_masker = T_audio.FrequencyMasking(freq_mask_param = self.freq_mask) if self.freq_mask > 0 else None



    def __len__(self):
        return len(self.files)

    def apply_augment(self, spec_1ch):
        """ spech1ch = tensor of shape (1, freq bins, time frames) """

        # RANDOM TIME SHIFT (CIRCULAR ROLL ALONG TIME AXIS)
        if self.time_shift > 0:
            shift = torch.randint(low = -self.time_shift, high = self.time_shift + 1, size = (1,)).item()
            spec_1ch = torch.roll(spec_1ch, shifts=shift, dims=2)

        # SPEC-AUGMENT MASKING
        if self.freq_masker is not None:
            spec_1ch = self.freq_masker(spec_1ch)
        if self.time_masker is not None:
            spec_1ch = self.time_masker(spec_1ch)
        
        # RETURN AUGMENTED SPECTROGRAM
        return spec_1ch
        

        

    def __getitem__(self, idx):
        path = self.files[idx]
        item = torch.load(path)
        
        spec = item['data']  # Already normalized to [0, 1] in preprocessing
        label = item['label']
        
        # Add channel dimension: (1, 224, 224)
        spec = spec.unsqueeze(0).float()
        
        if self.augment:
            spec = self.apply_augment(spec)
            
        # Convert to RGB (3 channels)
        spec = spec.repeat(3, 1, 1)
        
        # Apply ImageNet normalization
        spec = self.norm(spec)
        
        return spec, label
