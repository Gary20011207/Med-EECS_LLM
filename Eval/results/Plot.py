import pandas as pd
import matplotlib.pyplot as plt

# ----- 字型設定 -----
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",          # macOS
    "Microsoft JhengHei",        # Windows
    "Noto Sans TC",              # Linux
    "PingFang TC", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

# ----- 分類對應表 -----
CATEGORY_TRANSLATE = {
    "麻醉": "Anesthesia",
    "藥劑": "Pharmacy",
    "營養": "Nutrition",
    "復健": "Rehabilitation"
}

# ----- 讀取 CSV -----
CSV_PATH = "./Eval/results/Qwen2.5-14B-MiniLM-L6_no_rag_20250603_022028.csv"
df = pd.read_csv(CSV_PATH)

# ----- 類別轉英文 -----
df["category_en"] = df["category"].map(CATEGORY_TRANSLATE)

# ----- 可視化工具 -----
def plot_metric(df, metric_name: str, title_prefix: str):
    # 整體平均
    overall_mean = df[metric_name].mean()

    fig1, ax1 = plt.subplots(figsize=(4, 2))
    ax1.barh([metric_name], [overall_mean], color="#339af0")
    ax1.set_xlim(0, 1)
    ax1.set_title(f"{title_prefix} - Overall")
    ax1.set_xlabel(metric_name)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # 各分類平均
    cat_means = df.groupby("category_en")[metric_name].mean()

    fig2, ax2 = plt.subplots(figsize=(6, max(2, 0.4 * len(cat_means))))
    y = range(len(cat_means))
    ax2.barh(y, cat_means.values, color="#339af0")
    ax2.set_yticks(y)
    ax2.set_yticklabels(cat_means.index)
    ax2.set_xlim(0, 1)
    ax2.set_title(f"{title_prefix} - Per Category")
    ax2.set_xlabel(metric_name)
    ax2.invert_yaxis()
    for spine in ax2.spines.values():
        spine.set_visible(False)

    return fig1, fig2

# ----- 題目數量比例圖 -----
def plot_question_distribution(df):
    cat_counts = df["category_en"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 3))
    y = range(len(cat_counts))
    ax.barh(y, cat_counts.values, color="#845ef7")
    ax.set_yticks(y)
    ax.set_yticklabels(cat_counts.index)
    ax.set_title("Question Count per Category")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig

def plot_data_split():
    split_counts = {
        "Train (3-Fold)": 11,
        "Test": 3
    }

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(list(split_counts.keys()), list(split_counts.values()), color=["#228be6", "#fa5252"])
    ax.set_xlim(0, max(split_counts.values()) + 2)
    ax.set_title("VQA Benchmark Data Split")
    ax.set_xlabel("Number of Questions")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig


# ----- 產生圖表 -----
fig_tr_overall, fig_tr_cat = plot_metric(df, "token_recall", "Token Recall")
fig_bs_overall, fig_bs_cat = plot_metric(df, "bertscore_f1", "BERTScore F1")

# ----- 儲存或顯示 -----
fig_tr_overall.savefig("token_recall_overall.png", bbox_inches='tight')
fig_tr_cat.savefig("token_recall_by_category.png", bbox_inches='tight')
fig_bs_overall.savefig("bertscore_overall.png", bbox_inches='tight')
fig_bs_cat.savefig("bertscore_by_category.png", bbox_inches='tight')

fig_qcount = plot_question_distribution(df)
fig_qcount.savefig("question_distribution.png", bbox_inches='tight')

fig_split = plot_data_split()
fig_split.savefig("vqa_data_split.png", bbox_inches='tight')

print("✅ 圖片已成功輸出")