import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import os
import nibabel as nib
import re

# 디렉토리 경로 설정
t1c_bet_dir = "/home/iujeong/brain_meningioma/bet_output/bet_output_train/*_t1c_bet.nii.gz"
gtv_mask_dir = "/home/iujeong/brain_meningioma/bet_output/bet_output_train/*_gtv_mask.nii.gz"
original_npy_dir = "/home/iujeong/brain_meningioma/slice/original/slice_train/npy/*/*.npy"
padding_npy_dir = "/home/iujeong/brain_meningioma/slice/padded/slice_train_p/npy/*/*.npy"

# 랜덤 환자 선택 (공통 접두어로 매칭)
t1c_files = sorted(glob.glob(t1c_bet_dir))
gtv_files = sorted(glob.glob(gtv_mask_dir))
original_npy_files = sorted(glob.glob(original_npy_dir))
padding_npy_files = sorted(glob.glob(padding_npy_dir))

if not original_npy_files:
    raise FileNotFoundError("original_npy_files 경로에 파일이 없습니다. 경로를 다시 확인하세요.")
if not padding_npy_files:
    raise FileNotFoundError("padding_npy_files 경로에 파일이 없습니다. 경로를 다시 확인하세요.")

# 환자 ID 매칭을 위해 공통 prefix 사용
patient_ids = [os.path.basename(f).split('_t1c_bet')[0] for f in t1c_files]
rand_idx = random.randint(0, len(patient_ids)-1)
pid = patient_ids[rand_idx]

# 파일 경로 필터링
t1c_path = [f for f in t1c_files if pid in f][0]
gtv_path = [f for f in gtv_files if pid in f][0]

# 파일 로딩
t1c_img = nib.load(t1c_path).get_fdata()
gtv_img = nib.load(gtv_path).get_fdata()

# z축 가운데 슬라이스 기준으로 시각화
# GTV 마스크가 존재하는 z-slice를 찾기
nonzero_slices = np.any(gtv_img > 0, axis=(0, 1))
if not np.any(nonzero_slices):
    raise ValueError(f"{pid} 환자의 GTV 마스크에 종양이 없습니다.")

# 종양이 있는 z 중 가운데로 선택
valid_slices = np.where(nonzero_slices)[0]
mid_slice = valid_slices[len(valid_slices)//2]

# target_slice_filename 및 npy 파일 경로 결정
target_slice_filename = f"slice_{mid_slice}.npy"
original_npy_path = [f for f in original_npy_files if target_slice_filename in f and pid in f]
padding_npy_path = [f for f in padding_npy_files if target_slice_filename in f and pid in f]

if not original_npy_path:
    raise FileNotFoundError(f"{pid} 환자의 original npy 파일을 찾을 수 없습니다.")
if not padding_npy_path:
    raise FileNotFoundError(f"{pid} 환자의 padding npy 파일을 찾을 수 없습니다.")

original_npy = np.load(original_npy_path[0])
padding_npy = np.load(padding_npy_path[0])

t1c_slice = t1c_img[:, :, mid_slice]
gtv_slice = gtv_img[:, :, mid_slice]

# slice_idx는 더 이상 필요 없음, mid_slice 사용
if original_npy.ndim == 3:
    original_npy_slice = original_npy[:, :, mid_slice]
else:
    original_npy_slice = original_npy

if padding_npy.ndim == 3:
    padding_npy_slice = padding_npy[:, :, mid_slice]
else:
    padding_npy_slice = padding_npy

# 시각화
plt.figure(figsize=(20, 4))
plt.suptitle(f"Patient: {pid}", fontsize=16)

plt.subplot(1, 5, 1)
plt.imshow(t1c_slice, cmap='gray')
plt.title('T1c Skull-stripped')

plt.subplot(1, 5, 2)
plt.imshow(t1c_slice, cmap='gray')
plt.title('HD-BET')

plt.subplot(1, 5, 3)
plt.imshow(t1c_slice, cmap='gray')
plt.imshow(gtv_slice, cmap='Reds', alpha=0.4)
plt.title('BET + GTV Overlay')

plt.subplot(1, 5, 4)
plt.imshow(original_npy_slice, cmap='gray')
plt.title('Original NPY')

plt.subplot(1, 5, 5)
plt.imshow(padding_npy_slice, cmap='gray')
plt.title('Padding NPY')

plt.tight_layout()
save_dir = "/home/iujeong/brain_meningioma/visualize/slice_png_result"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, f'visual_{pid}_slice{mid_slice}.png'))
plt.show()