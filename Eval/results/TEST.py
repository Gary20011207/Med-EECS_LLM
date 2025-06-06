import pandas as pd

# 讀取 CSV 檔案
df = pd.read_csv("Qwen2.5-14B-e5-large_with_rag_20250602_144248.csv")  # ← 替換為你的實際檔名

# 計算 question + ground_truth 的字串總長度
df["q_gt_length"] = df["question"].astype(str).str.len() + df["ground_truth"].astype(str).str.len()

# 對每個 category 找出 q_gt_length 最短的那一筆
shortest_per_category = df.loc[df.groupby("category")["q_gt_length"].idxmin()]

# 輸出欄位
for _, row in shortest_per_category.iterrows():
    print(f"[Category: {row['category']}]")
    print("Question:", row["question"])
    print("Ground Truth:", row["ground_truth"])
    print("-" * 50)
