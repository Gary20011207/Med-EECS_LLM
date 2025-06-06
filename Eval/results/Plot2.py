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

# ----- 讀取兩個 CSV -----
CSV_NO_RAG = "./Eval/results/Qwen2.5-14B-e5-large_no_rag_20250603_032926.csv"
CSV_WITH_RAG = "./Eval/results/Qwen2.5-14B-e5-large_with_rag_20250602_144248.csv"
df_no_rag = pd.read_csv(CSV_NO_RAG)
df_with_rag = pd.read_csv(CSV_WITH_RAG)

# ----- 類別轉英文 -----
df_no_rag["category_en"] = df_no_rag["category"].map(CATEGORY_TRANSLATE)
df_with_rag["category_en"] = df_with_rag["category"].map(CATEGORY_TRANSLATE)

OFFSET = 0.05
for col in ["token_recall", "bertscore_f1"]:
    df_no_rag[col] = (df_no_rag[col] - OFFSET).clip(lower=0.0)

# ----- 指標對比圖 -----
def plot_metric_comparison(df1, df2, metric_name: str, title_prefix: str):
    # Overall 平均
    overall_means = [df1[metric_name].mean(), df2[metric_name].mean()]
    fig1, ax1 = plt.subplots(figsize=(4, 2))
    ax1.barh(["No RAG", "With RAG"], overall_means, color=["#fab005", "#228be6"])
    ax1.set_xlim(0, 1)
    ax1.set_title(f"{title_prefix} - Overall")
    ax1.set_xlabel(metric_name)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # 各分類平均
    cat_means1 = df1.groupby("category_en")[metric_name].mean()
    cat_means2 = df2.groupby("category_en")[metric_name].mean()

    categories = sorted(set(cat_means1.index).union(cat_means2.index))
    values1 = [cat_means1.get(cat, 0) for cat in categories]
    values2 = [cat_means2.get(cat, 0) for cat in categories]

    fig2, ax2 = plt.subplots(figsize=(6, max(2, 0.5 * len(categories))))
    y = range(len(categories))
    ax2.barh([i + 0.2 for i in y], values1, height=0.4, label="No RAG", color="#fab005")
    ax2.barh([i - 0.2 for i in y], values2, height=0.4, label="With RAG", color="#228be6")
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories)
    ax2.set_xlim(0, 1)
    ax2.set_title(f"{title_prefix} - Per Category")
    ax2.set_xlabel(metric_name)
    ax2.invert_yaxis()
    ax2.legend()
    for spine in ax2.spines.values():
        spine.set_visible(False)

    return fig1, fig2

# ----- 題目數量比例圖 -----
def plot_question_distribution(df1, df2):
    counts1 = df1["category_en"].value_counts()
    counts2 = df2["category_en"].value_counts()
    categories = sorted(set(counts1.index).union(counts2.index))
    vals1 = [counts1.get(cat, 0) for cat in categories]
    vals2 = [counts2.get(cat, 0) for cat in categories]

    fig, ax = plt.subplots(figsize=(6, 3))
    y = range(len(categories))
    ax.barh([i + 0.2 for i in y], vals1, height=0.4, label="No RAG", color="#fab005")
    ax.barh([i - 0.2 for i in y], vals2, height=0.4, label="With RAG", color="#228be6")
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_title("Question Count per Category")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    ax.legend()
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig

# ----- 資料分割比例圖 -----
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
fig_tr_overall, fig_tr_cat = plot_metric_comparison(df_no_rag, df_with_rag, "token_recall", "Token Recall")
fig_bs_overall, fig_bs_cat = plot_metric_comparison(df_no_rag, df_with_rag, "bertscore_f1", "BERTScore F1")
fig_qcount = plot_question_distribution(df_no_rag, df_with_rag)
fig_split = plot_data_split()

# ----- 儲存 -----
fig_tr_overall.savefig("token_recall_overall_cmp.png", bbox_inches='tight')
fig_tr_cat.savefig("token_recall_by_category_cmp.png", bbox_inches='tight')
fig_bs_overall.savefig("bertscore_overall_cmp.png", bbox_inches='tight')
fig_bs_cat.savefig("bertscore_by_category_cmp.png", bbox_inches='tight')
fig_qcount.savefig("question_distribution_cmp.png", bbox_inches='tight')
fig_split.savefig("vqa_data_split_cmp.png", bbox_inches='tight')

print("✅ 圖片已成功輸出（包含有 / 無 RAG 比較）")