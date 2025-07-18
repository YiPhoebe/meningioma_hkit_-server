import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# CSV 파일 경로 (수정해서 사용)
train_csv = "/home/iujeong/brain_meningioma/outputs/csv/slice_shape_stats.csv"
test_csv = "/home/iujeong/brain_meningioma/outputs/csv/slice_shape_stats_test.csv"

# 데이터 불러오기
df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# 데이터셋 구분
df_train["set"] = "train"
df_test["set"] = "test"
df_all = pd.concat([df_train, df_test], ignore_index=True)

# Height 분포
plt.figure(figsize=(10, 5))
sns.histplot(df_all["height"], bins=20, kde=True, color='skyblue')
plt.title("Height Distribution of Slices")
plt.xlabel("Height (pixels)")
plt.ylabel("Count")
plt.grid(True)
# Height 기준선
height_mean = df_all["height"].mean()
height_95 = df_all["height"].quantile(0.95)
plt.axvline(height_mean, color='blue', linestyle='--', label=f"Mean: {height_mean:.1f}")
plt.axvline(height_95, color='navy', linestyle=':', label=f"95%: {height_95:.1f}")
plt.xticks(range(120, 170, 5))
plt.legend()
plt.savefig("/home/iujeong/brain_meningioma/outputs/plt/height_distribution.png")
plt.close()

# Width 분포
plt.figure(figsize=(10, 5))
sns.histplot(df_all["width"], bins=20, kde=True, color='salmon')
plt.title("Width Distribution of Slices")
plt.xlabel("Width (pixels)")
plt.ylabel("Count")
plt.grid(True)
# Width 기준선
width_mean = df_all["width"].mean()
width_95 = df_all["width"].quantile(0.95)
plt.axvline(width_mean, color='red', linestyle='--', label=f"Mean: {width_mean:.1f}")
plt.axvline(width_95, color='darkred', linestyle=':', label=f"95%: {width_95:.1f}")
plt.xticks(range(150, 210, 5))
plt.legend()
plt.savefig("/home/iujeong/brain_meningioma/outputs/plt/width_distribution.png")
plt.close()