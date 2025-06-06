# ----- 匯入套件 -----
import pandas as pd
import matplotlib.pyplot as plt

# ----- 字型設定 -----
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Microsoft JhengHei", "Noto Sans TC",
    "PingFang TC", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["lines.linewidth"] = 0.75  # 線條粗細為原來一半

# ----- 分類對應表 -----
CATEGORY_TRANSLATE = {
    "麻醉": "Anesthesia",
    "藥劑": "Pharmacy",
    "營養": "Nutrition",
    "復健": "Rehabilitation"
}

# ----- 正確的 closed QA ID -----
correct_ids = {3, 6, 7, 8, 13, 18, 19, 20, 22}

# ----- 讀取單一 CSV 檔 -----
CSV_PATH = "./Eval/results/Qwen2.5-14B-e5-large_with_rag_20250602_144248.csv"
df = pd.read_csv(CSV_PATH)
df["category_en"] = df["category"].map(CATEGORY_TRANSLATE)

# ----- 過濾 closed 題型並標記正確性 -----
df_closed = df[df["type"] == "closed"].copy()
df_closed["is_correct"] = df_closed["qa_id"].apply(lambda x: x in correct_ids)

# ----- 計算每類別的正確率 -----
acc_by_category = df_closed.groupby("category_en")["is_correct"].mean()

# ----- 繪圖 -----
fig, ax = plt.subplots(figsize=(6, max(2, 0.5 * len(acc_by_category))))
y = range(len(acc_by_category))
ax.barh(y, acc_by_category.values, height=0.4, color="#51cf66")
ax.set_yticks(y)
ax.set_yticklabels(acc_by_category.index)
ax.set_xlim(0, 1)
ax.set_xlabel("Accuracy")
ax.set_title("Closed QA Accuracy per Category")
ax.invert_yaxis()
for spine in ax.spines.values():
    spine.set_visible(False)

# ----- 儲存圖檔 -----
fig.savefig("closed_accuracy_by_category.png", bbox_inches="tight")
print("✅ Closed QA Accuracy 圖已完成並輸出！")
