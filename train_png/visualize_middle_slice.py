import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# T1c와 GTV 오버레이를 시각화하여 PNG로 저장
t1c_dir = "/home/iujeong/brain_meningioma/raw_data/train_t1c"
gtv_dir = "/home/iujeong/brain_meningioma/raw_data/train_gtv"

# 출력 PNG 저장 경로
output_dir = "/home/iujeong/brain_meningioma/train_png"
os.makedirs(output_dir, exist_ok=True)

for f in sorted(os.listdir(t1c_dir)):
    if not f.endswith(".nii.gz"):
        continue

    t1c_path = os.path.join(t1c_dir, f)
    case_id = f.replace("_t1c.nii.gz", "")
    gtv_name = case_id + "_gtv.nii.gz"
    gtv_path = os.path.join(gtv_dir, gtv_name)

    if not os.path.exists(gtv_path):
        print(f"[!] GTV not found for: {case_id}")
        continue

    try:
        t1c_img = nib.load(t1c_path).get_fdata()
        gtv_img = nib.load(gtv_path).get_fdata()

        gtv_sums = gtv_img.sum(axis=(0, 1))
        if np.max(gtv_sums) == 0:
            print(f"[!] No GTV found in any slice for: {case_id}")
            continue
        max_idx = np.argmax(gtv_sums)

        t1c_slice = t1c_img[:, :, max_idx]
        gtv_slice = gtv_img[:, :, max_idx]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(t1c_slice.T, cmap="gray", origin="lower")
        plt.title(f"{case_id} T1c")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(t1c_slice.T, cmap="gray", origin="lower")
        plt.imshow(gtv_slice.T, cmap="Reds", alpha=0.4, origin="lower")
        plt.title(f"{case_id} + GTV")
        plt.axis("off")

        save_path = os.path.join(output_dir, f"{case_id}_gtv_overlay.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved: {save_path}")
        print(f"[i] Max GTV slice index for {case_id}: {max_idx}, sum={gtv_sums[max_idx]}")

    except Exception as e:
        print(f"[!] Error with {case_id}: {e}")