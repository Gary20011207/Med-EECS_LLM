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
plt.rcParams["lines.linewidth"] = 0.75

# ----- 類別對應表 -----
CATEGORY_TRANSLATE = {
    "麻醉": "Anesthesia",
    "藥劑": "Pharmacy",
    "營養": "Nutrition",
    "復健": "Rehabilitation"
}

# ----- 載入資料 -----
CSV_PATH = "./Eval/results/Qwen2.5-14B-e5-large_with_rag_20250602_144248.csv"  # 只讀取一個檔案
df = pd.read_csv(CSV_PATH)
df["category_en"] = df["category"].map(CATEGORY_TRANSLATE)

# ----- 計算每個類別下 open/closed 的比例 -----
type_counts = df.groupby(["category_en", "type"]).size().unstack(fill_value=0)
type_ratio = type_counts.div(type_counts.sum(axis=1), axis=0)  # 轉為比例

# ----- 繪圖 -----
fig, ax = plt.subplots(figsize=(8, 0.8 * len(type_counts)))  # 拉高 Y 軸間距
fig.subplots_adjust(bottom=0.25)  # 留空間給 legend
y = range(len(type_counts))

# 取得各分類的 open / closed 數量
opens = type_counts["open"] if "open" in type_counts else [0] * len(type_counts)
closeds = type_counts["closed"] if "closed" in type_counts else [0] * len(type_counts)
categories = type_counts.index.tolist()

# 每列兩條 bar：closed 向左偏移、open 向右偏移
bar_height = 0.4
ax.barh([i + 0.2 for i in y], opens, height=bar_height, label="Open", color="#4dabf7")
ax.barh([i - 0.2 for i in y], closeds, height=bar_height, label="Closed", color="#82c91e")

ax.set_yticks(y)
ax.set_yticklabels(categories)
ax.set_xlabel("Number of Questions")
ax.set_title("Number of Open/Closed Questions per Category")
ax.invert_yaxis()

# Legend 放在下方中央
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False
)

for spine in ax.spines.values():
    spine.set_visible(False)

# ----- 儲存 -----
fig.savefig("question_type_count_by_category.png", bbox_inches="tight")
print("✅ 題型數量圖已完成並輸出！")

