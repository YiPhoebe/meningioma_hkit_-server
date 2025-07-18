
from glob import glob
import os

for phase in ["norm_test", "norm_train", "norm_valid"]:
    path = f"/home/iujeong/brain_meningioma/prepocessing/normalized_volume/{phase}"
    files = glob(os.path.join(path, "*_norm.nii.gz"))
    patient_ids = set(os.path.basename(f).replace("_norm.nii.gz", "") for f in files)
    print(f"{phase}: {len(patient_ids)}명")