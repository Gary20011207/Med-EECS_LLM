import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import tempfile

# ------------ 載入 CSV ------------
CSV_PATH = "./Eval/results/COMPARE.csv"
df = pd.read_csv(CSV_PATH)
TOTAL = len(df)

# ------------ 初始化狀態 ----------
def init_state():
    return {"index": 0, "results": []}

# ------------ 繪圖工具 ------------
# ------------ 繪圖工具 ------------
def build_figures(df_res: pd.DataFrame):
    # --- 1️⃣ 中文類別轉英文 ---
    category_trans = {
        "麻醉": "Anesthesia",
        "藥劑": "Pharmacy",
        "營養": "Nutrition",
        "復健": "Rehabilitation"
    }
    df_res["category_en"] = df_res["category"].map(category_trans).fillna(df_res["category"])

    # --- 2️⃣ 字型設定（避免亂碼） ---
    plt.rcParams["font.family"] = "DejaVu Sans"  # 英文專用無亂碼
    plt.rcParams["axes.unicode_minus"] = False

    # --- 3️⃣ 整體比例圖 ---
    overall_cnt = df_res["review"].value_counts().reindex(["好", "不好"], fill_value=0)
    overall_prop = overall_cnt / overall_cnt.sum()

    fig1, ax1 = plt.subplots(figsize=(10, 2))
    ax1.barh(["Overall"], [overall_prop["好"]], color="#37b24d")
    ax1.barh(["Overall"], [overall_prop["不好"]], color="#e03131",
             left=[overall_prop["好"]])
    ax1.set_xlim(0, 1); ax1.axis("off")

    # --- 4️⃣ 各分類比例圖 ---
    cat_grp = (df_res.groupby(["category_en", "review"])
                         .size()
                         .unstack(fill_value=0)
                         .reindex(columns=["好", "不好"]))
    cat_prop = cat_grp.div(cat_grp.sum(axis=1), axis=0)

    fig2, ax2 = plt.subplots(figsize=(10, max(2, 0.4*len(cat_prop))))
    y = range(len(cat_prop))
    ax2.barh(y, cat_prop["好"], color="#37b24d")
    ax2.barh(y, cat_prop["不好"], color="#e03131",
             left=cat_prop["好"])
    ax2.set_yticks(y); ax2.set_yticklabels(cat_prop.index)
    ax2.set_xlim(0, 1); ax2.invert_yaxis()
    for spine in ax2.spines.values():
        spine.set_visible(False)

    return fig1, fig2

# ------------ 產生 Markdown / 進度 ------------
def render(idx: int):
    if idx >= TOTAL:
        return "## 🎉 所有題目已完成！", "", 1.0, "100 %"
    row = df.iloc[idx]
    ques_md = f"""
### 題目 {idx+1} / {TOTAL}

**分類：** `{row['category']} / {row['sub_category']}`

**問題：**  
{row['question']}
"""
    ans_html = f"""
<div class='answer-box'>
<pre>{row['model_answer']}</pre>
</div>
"""
    pct_txt = f"{int(idx/TOTAL*100):>3d} %"
    return ques_md, ans_html, idx / TOTAL, pct_txt

# ------------ 評分回調 ------------
def rate(state, choice):
    idx = state["index"]
    if idx < TOTAL:
        state["results"].append({**df.iloc[idx].to_dict(), "review": choice})
        state["index"] += 1

    ques_md, ans_html, prog_val, pct_txt = render(state["index"])

    # 預設 hidden
    overall_upd = gr.update(visible=False)
    cat_upd     = gr.update(visible=False)
    dl_upd      = gr.update(visible=False)

    # 若已完成 → 產生圖 + 下載檔 & 顯示
    if state["index"] >= TOTAL:
        df_res = pd.DataFrame(state["results"])
        fig1, fig2 = build_figures(df_res)

        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_res.to_csv(tmp_csv.name, index=False)

        overall_upd = gr.update(value=fig1, visible=True)
        cat_upd     = gr.update(value=fig2, visible=True)
        dl_upd      = gr.update(value=tmp_csv.name, visible=True)

    return (ques_md, ans_html,
            gr.update(value=prog_val), pct_txt,
            overall_upd, cat_upd, dl_upd,
            state)

# ------------ Gradio 介面 ------------
with gr.Blocks(css="""
body {background:#eceff3;font-family:'Helvetica Neue',Arial,sans-serif;color:#1f1f1f}
.gr-button {font-size:20px !important;height:3em !important}
input[type=range]{height:28px}

/* 模型回答區：純白底、黑字、固定高度 + 捲軸 */
.answer-box {
    background:#000000;          /* 白底 */
    color:#000000;               /* 黑字 */
    border:1px solid #d0d0d0;    /* 淡灰邊框 */
    border-radius:8px;
    height:350px;
    overflow-y:auto;
    padding:16px;
    font-family:Courier New,monospace;
    white-space:pre-wrap;
    box-shadow:0 2px 6px rgba(0,0,0,.08);
}

/* 固定底部按鈕列 */
#btnrow {
    position:fixed;
    bottom:20px;
    left:50%;
    transform:translateX(-50%);
    width:90%;
    justify-content:center;
    gap:24px;
    z-index:999;
}
""") as demo:
    gr.Markdown("## 🧪 ERAS QA 模型評估介面")

    progress_bar = gr.Slider(minimum=0, maximum=1, value=0,
                             interactive=False, show_label=False)
    progress_pct = gr.Markdown("0 %")

    ques_md = gr.Markdown()
    ans_html = gr.HTML()

    with gr.Row(elem_id="btnrow"):
        good_btn = gr.Button("✅ 良好", variant="primary")
        bad_btn  = gr.Button("❌ 不好", variant="stop")

    plot_overall = gr.Plot(label="整體比例", visible=False)
    plot_cat     = gr.Plot(label="各分類比例", visible=False)
    download_file = gr.File(label="下載結果", visible=False)

    hidden_state = gr.State(init_state())

    # 初始畫面
    q0, a0, p0, pct0 = render(0)
    ques_md.value, ans_html.value, progress_bar.value, progress_pct.value = q0, a0, p0, pct0

    # 共同輸出序列
    outputs = [ques_md, ans_html, progress_bar, progress_pct,
               plot_overall, plot_cat, download_file, hidden_state]

    good_btn.click(lambda s: rate(s, "好"),  inputs=hidden_state, outputs=outputs)
    bad_btn.click( lambda s: rate(s, "不好"), inputs=hidden_state, outputs=outputs)

# ------------ 啟動 ------------
if __name__ == "__main__":
    demo.launch(share=True)
