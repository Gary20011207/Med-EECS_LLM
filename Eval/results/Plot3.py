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
CSV_REVIEW = "./Eval/results/Expert_Eval.csv"
df = pd.read_csv(CSV_REVIEW)
df["category_en"] = df["category"].map(CATEGORY_TRANSLATE)

# ----- 評分轉換函式 -----
def review_to_ratio(text):
    if isinstance(text, str):
        good = text.count("好")
        bad = text.count("不好")
        total = good + bad
        return good / total if total > 0 else None
    return None

df["review_score"] = df["review"].apply(review_to_ratio)

# ----- 繪圖函式 -----
def plot_review_ratio(df, title_prefix="Good Review Ratio"):
    # 整體好評比例
    overall_ratio = df["review_score"].mean()
    fig1, ax1 = plt.subplots(figsize=(4, 2))
    ax1.barh(["All"], [overall_ratio], color="#40c057")
    ax1.set_xlim(0, 1)
    ax1.set_title(f"{title_prefix} - Overall")
    ax1.set_xlabel("Ratio")
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # 各分類好評比例
    cat_ratio = df.groupby("category_en")["review_score"].mean()
    categories = cat_ratio.index.tolist()
    values = cat_ratio.values

    fig2, ax2 = plt.subplots(figsize=(6, max(2, 0.5 * len(categories))))
    y = range(len(categories))
    ax2.barh(y, values, height=0.4, color="#40c057")
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories)
    ax2.set_xlim(0, 1)
    ax2.set_title(f"{title_prefix} - Per Category")
    ax2.set_xlabel("Ratio")
    ax2.invert_yaxis()
    for spine in ax2.spines.values():
        spine.set_visible(False)

    return fig1, fig2

# ----- 繪製並儲存圖表 -----
fig_ratio_overall, fig_ratio_cat = plot_review_ratio(df)
fig_ratio_overall.savefig("review_ratio_overall.png", bbox_inches='tight')
fig_ratio_cat.savefig("review_ratio_by_category.png", bbox_inches='tight')

print("✅ 好評比例圖已完成並輸出！")
