

import os
import pandas as pd
from glob import glob

# 경로 정의
base_dir = "/home/iujeong/brain_meningioma/prepocessing/normalized_volume"
split_dirs = {
    "train": os.path.join(base_dir, "norm_train"),
    "valid": os.path.join(base_dir, "norm_valid"),
    "test": os.path.join(base_dir, "norm_test"),
}

# 저장 경로 (base_dir로 저장)
save_dir = base_dir

# 각 폴더 처리
for split, dir_path in split_dirs.items():
    file_list = sorted(glob(os.path.join(dir_path, "*.nii.gz")))
    file_names = [os.path.basename(f) for f in file_list]
    
    df = pd.DataFrame(file_names, columns=["filename"])
    save_path = os.path.join(save_dir, f"{split}_files.csv")
    df.to_csv(save_path, index=False)
    print(f"{split} → {len(file_names)}개 파일 저장 완료: {save_path}")