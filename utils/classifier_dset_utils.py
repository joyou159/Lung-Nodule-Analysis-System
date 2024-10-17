import torch 
import random
from common_utils.util import *
from common_utils.disk_caching import * 
import numpy as np 
import torch.nn as nn 
from CT import get_ct, get_ct_raw_candidates


class AugmentationCandidate(nn.Module):
    def __init__(self, augmentation_dict):
        super().__init__()

        self.flip = augmentation_dict.flip 
        self.offset = augmentation_dict.offset 
        self.scale = augmentation_dict.scale 
        self.rotate = augmentation_dict.rotate
        self.noise = augmentation_dict.noise 


    def forward(self, series_uid, center_xyz, width_irc, use_cache = True):

        if use_cache:
            ct_chunks, center_irc = get_ct_raw_candidates(series_uid, center_xyz, width_irc) 
        else:
            ct = get_ct(series_uid)
            ct_chunks, center_irc = ct.get_raw_candidate_nodule(center_xyz, width_irc)

        ct_tensor = torch.tensor(ct_chunks).unsqueeze(0).unsqueeze(0).to("cuda",torch.float32) # the (batch_size, channel, depth, heigth, width)  the expected shape of pytorch 
        
        transform_mat = self.build_3d_transformation_matrix()
        transform_mat = transform_mat.to(ct_tensor.device, torch.float32) # loading the matrix to GPU
        
        affine_transform = nn.functional.affine_grid(transform_mat[:3].unsqueeze(0), # transformation_mat[:3] coordinates only (not density value) 
                                                 ct_tensor.size(), align_corners=False)
    
            
        augmented_chunk = nn.functional.grid_sample(ct_tensor, affine_transform, padding_mode="border", align_corners=False)

        if self.noise:
            noise_added = torch.rand_like(augmented_chunk) * self.noise
            augmented_chunk += noise_added

        return augmented_chunk[0], center_irc

    def build_3d_transformation_matrix(self):
        transformation_mat = torch.eye(4) # start off (just identity)

        for i in range(3): # per axis 
            if self.flip:
                if random.random() > 0.3: # flipping is a bit random 
                    transformation_mat[i,i] *= -1

            if self.offset:
                offset_value = self.offset # must be limited [-1,1]
                random_factor = (random.random() * 2 - 1)  # (std -> 2) and (mean -> -1)
                transformation_mat[i,3] = offset_value * random_factor # the grid_sample will interpolate since the transition won't be in voxel steps 

            if self.scale:
                scaling_value = self.scale
                random_factor = (random.random() * 2 - 1)
                transformation_mat[i,i] *= 1 + scaling_value * random_factor
    
        # rotation around z-axis (because the scale of this axis is completely different from those of x & y)
        if self.rotate:
            angle_in_rad = random.random() * np.pi * 2
            s = np.sin(angle_in_rad)
            c = np.cos(angle_in_rad)

            rotation_mat = torch.tensor(
                [
                    [c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]
            ) 

            transformation_mat @= rotation_mat.to(torch.float32) # accumulate the transformation matrices

    
        return transformation_mat

def augmentation_3D(input_batch, augmentation_dict):
    
    transform_t = torch.eye(4, dtype=torch.float32)
    for i in range(3):
        if augmentation_dict["flip"]: 
            if random.random() > 0.5:
                transform_t[i,i] *= -1
                
        if augmentation_dict["offset"]: 
            offset_float = augmentation_dict["offset"]
            random_float = (random.random() * 2 - 1)
            transform_t[3,i] = offset_float * random_float
            
        if augmentation_dict["scale"]:
            scaling_value = augmentation_dict["scale"]
            random_factor = (random.random() * 2 - 1)
            transform_t[i,i] *= 1 + scaling_value * random_factor
            
    if augmentation_dict["rotate"]:
        angle_rad = random.random() * np.pi * 2
        s = np.sin(angle_rad)
        c = np.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        transform_t @= rotation_t
        
    affine_t = torch.nn.functional.affine_grid(
            transform_t[:3].unsqueeze(0).expand(input_batch.size(0), -1, -1).to(input_batch.device),
            input_batch.shape,
            align_corners=False,
        )

    augmented_chunk = torch.nn.functional.grid_sample(
            input_batch,
            affine_t,
            padding_mode='border',
            align_corners=False,
        )
    if augmentation_dict["noise"]:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']
        augmented_chunk += noise_t
        
    return augmented_chunk


