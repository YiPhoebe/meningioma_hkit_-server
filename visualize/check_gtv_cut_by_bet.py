import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def check_gtv_vs_hd_bet(t1c_path, brain_mask_path, gtv_mask_path):
    img = nib.load(t1c_path).get_fdata()
    brain_mask = nib.load(brain_mask_path).get_fdata()
    print("brain_mask stats → min:", np.min(brain_mask), "max:", np.max(brain_mask), "unique:", np.unique(brain_mask))
    gtv_mask = nib.load(gtv_mask_path).get_fdata()

    lost_voxels = np.logical_and(gtv_mask == 1, brain_mask == 0)
    lost_count = np.sum(lost_voxels)
    print(f"[{t1c_path}] 잘린 GTV voxel 수: {lost_count}")

    gtv_mask_bin = gtv_mask > 0
    gtv_z_indices = np.where(np.sum(gtv_mask_bin, axis=(0, 1)) > 0)[0]

    for z in gtv_z_indices:
        gtv_slice = gtv_mask[:, :, z]
        brain_slice = brain_mask[:, :, z]
        
        gtv_bin = gtv_slice >= 0.5
        brain_bin = brain_slice >= 0.5
        n_cut_voxels = np.sum(np.logical_and(gtv_bin, ~brain_bin))
        n_gtv_voxels = np.sum(gtv_bin)
        cut_ratio = n_cut_voxels / (n_gtv_voxels + 1e-5)
        print(f"{os.path.basename(t1c_path)} z={z} → GTV={n_gtv_voxels}, cut={n_cut_voxels}, ratio={cut_ratio:.3f}")
        if cut_ratio > 0.10:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img[:, :, z], cmap='gray')
            axs[0].set_title(f"Original z={z}")
            axs[1].imshow(img[:, :, z] * brain_mask[:, :, z], cmap='gray')
            axs[1].set_title("After HD-BET")
            axs[2].imshow(img[:, :, z], cmap='gray')
            axs[2].imshow(gtv_mask[:, :, z], cmap='Reds', alpha=0.4)
            axs[2].set_title("Overlay: GTV")

            fig.suptitle(f"Cut ratio: {cut_ratio:.1%}", fontsize=14, color='red')
            plt.tight_layout()
            save_path = f"/home/iujeong/brain_meningioma/visualize/check_gtv_cut_by_bet/fig_{os.path.basename(t1c_path).replace('.nii.gz','')}_z{z}.png"
            plt.savefig(save_path)
            print(f"Saved figure: {save_path}")
            plt.close()


# # --- Batch lost voxel check and log to CSV ---
import glob
import pandas as pd

t1c_dir = "/home/iujeong/brain_meningioma/raw_data/resampled/all_t1c/"
gtv_dir = "/home/iujeong/brain_meningioma/raw_data/resampled/all_gtv/"
mask_dir = "/home/iujeong/brain_meningioma/bet_output/bet_output_train/"
log_path = "/home/iujeong/brain_meningioma/visualize/check_gtv_cut_by_bet/lost_voxel_log.csv"

results = []

for t1c_path in sorted(glob.glob(os.path.join(t1c_dir, "*.nii.gz"))):
    case_id = os.path.basename(t1c_path).replace("_t1c.nii.gz", "")
    mask_path = os.path.join(mask_dir, f"{case_id}_t1c_bet_mask.nii.gz")
    gtv_path = os.path.join(gtv_dir, f"{case_id}_gtv.nii.gz")

    if not (os.path.exists(t1c_path) and os.path.exists(mask_path) and os.path.exists(gtv_path)):
        print(f"Skipping {case_id}: missing file")
        continue

    try:
        img = nib.load(t1c_path).get_fdata()
        brain_mask = nib.load(mask_path).get_fdata()
        gtv_mask = nib.load(gtv_path).get_fdata()

        lost_voxels = np.logical_and(gtv_mask == 1, brain_mask == 0)
        lost_count = int(np.sum(lost_voxels))
        print(f"{case_id} → lost_voxels: {lost_count}")
        check_gtv_vs_hd_bet(t1c_path, mask_path, gtv_path)
        results.append({"case_id": case_id, "lost_voxels": lost_count})
    except Exception as e:
        print(f"Error processing {case_id}: {e}")

df = pd.DataFrame(results)
df.to_csv(log_path, index=False)
print(f"\nLog saved to: {log_path}")