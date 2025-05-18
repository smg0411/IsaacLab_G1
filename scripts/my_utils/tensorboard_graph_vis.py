import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp



# CSV 파일 경로
csv_path = r"C:\IsaacLab\logs\rsl_rl\g1_lift\2025-05-11_11-33-59_G1_Lift_v14\tensorboard\reach.csv"  # 경로는 실제 파일 위치에 따라 수정

outdir = osp.dirname(csv_path)

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 그래프 시각화 (점 없이 선만 그리기)
plt.figure(figsize=(10, 5))
plt.plot(df["Step"], df["Value"], linestyle='-', linewidth=1)
plt.title("")
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.grid(True)
plt.tight_layout()

save_path = osp.join(outdir, "reach.png")
plt.savefig(save_path)

plt.show()
