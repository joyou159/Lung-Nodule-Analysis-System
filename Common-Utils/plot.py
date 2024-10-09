import matplotlib.pyplot as plt
import numpy as np 
from util import CandidateInfoTuple, IRC_tuple
from CT import  get_ct


def visualize_candidate(ct_candidate): 
    """
    Show the whole CT scan the accompanied nodule candidate.
    
    Parameters:
        ct_candidate (CandidateInfoTuple): information list of that candidate nodule.
    """
    subject_ct = get_ct(ct_candidate.series_uid)
    irc_diameters = IRC_tuple(32, 48, 48) 
    ct_chunks, icr_center = subject_ct.get_raw_candidate_nodule(ct_candidate.center_xyz,irc_diameters)
    
    if ct_candidate.diameter_mm: 
        radius = round(ct_candidate.diameter_mm)
        color = 'b'
        label = "Positive"
    else:
        radius = 10 
        color = 'r' # means missing 
        label = "Positive, Yet Non-annotated," if ct_candidate.isNodule_bool else "Negative"
        
        
    fig_1, axes_1 = plt.subplots(1, 3, figsize=(15, 5))
    fig_1.suptitle(f"The Entire CT Scan, Besides the {label} Nodule")
    axes_1[0].imshow(subject_ct.hu_arr[icr_center[0], :,:], cmap='gray')
    index_circle = plt.Circle((icr_center[2],icr_center[1]), radius=radius, color=color, fill=False, linewidth=2)  # 'b' stands for blue color
    axes_1[0].add_patch(index_circle)
    axes_1[0].set_title('Axial Slice (Index)')
    axes_1[0].axis("off")

    axes_1[1].imshow(subject_ct.hu_arr[:,icr_center[1], :], cmap='gray')
    coronal_circle = plt.Circle((icr_center[2],icr_center[0]), radius=radius, color=color, fill=False, linewidth=2)  # 'b' stands for blue color
    axes_1[1].add_patch(coronal_circle)
    axes_1[1].set_title('Coronal Slice (Row)')
    axes_1[1].invert_yaxis()     
    axes_1[1].axis("off")

    axes_1[2].imshow(subject_ct.hu_arr[:, :, icr_center[2]], cmap='gray')
    sagittal_circle = plt.Circle((icr_center[1],icr_center[0]), radius=radius, color=color, fill=False, linewidth=2)  # 'b' stands for blue color
    axes_1[2].add_patch(sagittal_circle)
    axes_1[2].set_title('Sagittal Slice (Column)')
    axes_1[2].invert_yaxis()
    axes_1[2].axis("off")

    fig_2, axes_2 = plt.subplots(1, 3, figsize=(15, 5))
    fig_2.suptitle(f"Magnified View of the {label} Nodule")
    axes_2[0].imshow(ct_chunks[ct_chunks.shape[0]//2, :,:], cmap='gray')
    axes_2[0].set_title('Axial Slice (Index)')
    axes_2[0].axis("off")

    axes_2[1].imshow(ct_chunks[:,ct_chunks.shape[1]//2, :], cmap='gray')
    axes_2[1].set_title('Coronal Slice (Row)')
    axes_2[1].invert_yaxis()
    axes_2[1].axis("off")

    axes_2[2].imshow(ct_chunks[:, :, ct_chunks.shape[2]//2], cmap='gray')
    axes_2[2].set_title('Sagittal Slice (Column)')
    axes_2[2].invert_yaxis()
    axes_2[2].axis("off")
    
    print(f"The candidate information: {ct_candidate} \n")
    print(f"shape of CT scan {subject_ct.hu_arr.shape}")
    print(f"The exact location of candidate nodule {icr_center}")



def visualize_ct_and_mask(ct_slice, mask):
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display CT slice alone
    axs[0].imshow(ct_slice, cmap='gray')
    axs[0].set_title('CT Slice')
    axs[0].axis('off') 

    # Create a color image for the superimposed mask
    colored_mask = np.zeros((*mask.shape, 3)) 
    colored_mask[mask == 1] = [1, 0, 0]  # Red for mask areas

    # Display CT slice with the mask superimposed
    axs[1].imshow(ct_slice, cmap='gray')
    axs[1].imshow(colored_mask, alpha=0.5)  
    axs[1].set_title('CT Slice with Mask')
    axs[1].axis('off') 
    
    plt.tight_layout()
