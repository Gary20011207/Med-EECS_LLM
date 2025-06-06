import pandas as pd
import matplotlib.pyplot as plt

# ----- 資料定義 -----

# 有 LoRA 的結果
df_lora = pd.DataFrame({
    "ID": ["sub007", "sub010", "sub013"],
    "Token-Level Recall": [0.2687, 0.4848, 0.4516],
    "BERTScore_F1": [0.8311, 0.8418, 0.8605],
    "Source": "With LoRA"
})

# 無 LoRA 的結果
df_no_lora = pd.DataFrame({
    "ID": ["sub007", "sub010", "sub013"],
    "Token-Level Recall": [0.2388, 0.4848, 0.2903],
    "BERTScore_F1": [0.8141, 0.8562, 0.8148],
    "Source": "No LoRA"
})

# 合併資料
df_all = pd.concat([df_lora, df_no_lora], ignore_index=True)

# ----- 字型設定 -----
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Microsoft JhengHei", "Noto Sans TC",
    "PingFang TC", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["lines.linewidth"] = 0.75

# ----- 每題雙指標比較圖 -----
ids = df_all["ID"].unique()
y = range(len(ids))

# ----- 單獨：Token-Level Recall 比較圖 -----
fig_token, ax_token = plt.subplots(figsize=(9, 2.5))
vals_with_lora = df_all[(df_all["Source"] == "With LoRA")]["Token-Level Recall"].values
vals_no_lora = df_all[(df_all["Source"] == "No LoRA")]["Token-Level Recall"].values
ax_token.barh([i + 0.2 for i in y], vals_with_lora, height=0.4, label="With LoRA", color="#fab005")
ax_token.barh([i - 0.2 for i in y], vals_no_lora, height=0.4, label="No LoRA", color="#ffd43b")
ax_token.set_yticks(y)
ax_token.set_yticklabels(ids)
ax_token.set_xlim(0, 1)
ax_token.set_xlabel("Token-Level Recall")
ax_token.set_title("Token-Level Recall: With vs Without LoRA")
ax_token.invert_yaxis()
ax_token.legend()
for spine in ax_token.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.show()

# ----- 單獨：BERTScore F1 比較圖 -----
fig_bert, ax_bert = plt.subplots(figsize=(9, 2.5))
vals_with_lora = df_all[(df_all["Source"] == "With LoRA")]["BERTScore_F1"].values
vals_no_lora = df_all[(df_all["Source"] == "No LoRA")]["BERTScore_F1"].values
ax_bert.barh([i + 0.2 for i in y], vals_with_lora, height=0.4, label="With LoRA", color="#228be6")
ax_bert.barh([i - 0.2 for i in y], vals_no_lora, height=0.4, label="No LoRA", color="#74c0fc")
ax_bert.set_yticks(y)
ax_bert.set_yticklabels(ids)
ax_bert.set_xlim(0, 1)
ax_bert.set_xlabel("BERTScore F1")
ax_bert.set_title("BERTScore F1: With vs Without LoRA")
ax_bert.invert_yaxis()
ax_bert.legend()
for spine in ax_bert.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.show()