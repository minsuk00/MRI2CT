import os
import torch
import torchio as tio
from src.utils import anatomix_normalize

class DataPreprocessing(tio.Transform):
    def __init__(self, patch_size=96, enable_safety_padding=False, res_mult=32, use_weighted_sampler=False, **kwargs):        
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.enable_safety_padding = enable_safety_padding
        self.res_mult = res_mult
        self.use_weighted_sampler = use_weighted_sampler

    def apply_transform(self, subject):
        # subject['ct'].set_data(anatomix_normalize(subject['ct'].data, clip_range=(-450, 450)).float())
        subject['ct'].set_data(anatomix_normalize(subject['ct'].data, clip_range=(-1024, 1024)).float())
        # subject['mri'].set_data(anatomix_normalize(subject['mri'].data, percentile_range=(0,99.99)).float())
        subject['mri'].set_data(anatomix_normalize(subject['mri'].data).float())
        
        # Save original shape
        subject['original_shape'] = torch.tensor(subject['ct'].spatial_shape)
        
        pad_offset=0
        # Padding logic
        if self.enable_safety_padding:
            pad_val = self.patch_size//2
            subject = tio.Pad(pad_val, padding_mode=0)(subject)
            pad_offset=pad_val

        subject['pad_offset'] = pad_offset

        current_shape = subject['ct'].spatial_shape
        padding_params = []
        for dim in current_shape:
            target = max(self.patch_size, (int(dim) + self.res_mult - 1) // self.res_mult * self.res_mult)
            padding_params.extend([0, int(target - dim)])
            
        if any(p > 0 for p in padding_params):
            subject = tio.Pad(padding_params, padding_mode=0)(subject)
            
        # Probability Map for Sampler
        if self.use_weighted_sampler and 'prob_map' not in subject:
            prob = (subject['ct'].data > 0.01).to(torch.float32)
            subject.add_image(tio.LabelMap(tensor=prob, affine=subject['mri'].affine), 'prob_map')

        spatial_shape = subject['mri'].spatial_shape
        if any(d % self.res_mult != 0 for d in spatial_shape):
             print(f"[WARNING] Volume shape {spatial_shape} is not a multiple of {self.res_mult}!")
            
        return subject

def get_augmentations():
    return tio.Compose([
        # Applied to BOTH MRI and CT identically
        tio.OneOf({
            tio.RandomElasticDeformation(
                num_control_points=7, 
                max_displacement=4, 
                locked_borders=2, 
                image_interpolation='linear' 
            ): 0.3, 
            tio.RandomAffine(
                scales=(0.95, 1.1), 
                degrees=7, 
                translation=4,
                default_pad_value='minimum'
            ): 0.7,
        }, p=0.8), 
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.Clamp(0, 1),
        
        tio.Compose([
            tio.RandomBiasField(coefficients=0.5, p=0.4, order=2),
            tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.4),
        ], include=['mri']) ,
        tio.Clamp(0, 1),
    ])

def get_subject_paths(root, relative_path):
    """
    root: base directory (e.g., .../3.0x3.0x3.0mm)
    relative_path: 'train/1ABA005' or just '1ABA005' if using flat structure
    """
    # Construct full path
    subj_dir = os.path.join(root, relative_path)
    
    ct_path = os.path.join(subj_dir, "ct.nii.gz")
    mr_path = os.path.join(subj_dir, "registration_output", "moved_mr.nii.gz")
    
    # Fallback for checking existence
    if not os.path.exists(ct_path) or not os.path.exists(mr_path):
        raise FileNotFoundError(f"Missing files in {subj_dir}")
        
    return {'ct': ct_path, 'mri': mr_path}
