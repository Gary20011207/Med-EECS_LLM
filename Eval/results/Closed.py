import pandas as pd

# ----- 讀取 CSV -----
CSV_PATH = "./Eval/results/Qwen2.5-14B-e5-large_with_rag_20250602_144248.csv"  # 替換為你的實際檔案路徑
df = pd.read_csv(CSV_PATH)

# ----- 篩選 type = "closed" 的資料 -----
df_closed = df[df["type"] == "closed"]

# ----- 取出指定欄位 -----
df_selected = df_closed[["qa_id", "model_answer", "ground_truth"]]

# ----- 逐筆格式化輸出 -----
for idx, row in df_selected.iterrows():
    print(f"📌 QA ID: {row['qa_id']}")
    print(f"🤖 Model Answer:\n{row['model_answer']}")
    print(f"✅ Ground Truth:\n{row['ground_truth']}")
    print("-" * 50)