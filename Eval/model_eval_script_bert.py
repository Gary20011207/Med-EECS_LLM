#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERAS QA 模型評估腳本（新版）
評估不同模型配置在 ERAS QA 資料集上的表現，
支援中文 BERTScore 與 Token-level Recall。
"""

import os
import time
import json
import re
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from bert_score import score as bertscore

# ---------------- 專案核心模組 ----------------
from core.model_manager import ModelManager
from core.db_manager import DBManager
from core.rag_engine import RAGEngine
# ------------------------------------------------

# ---------- 日誌設定 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"model_eval_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("model_eval")
# -----------------------------

# ---------- 公用函數 ----------
_tok_split_en = re.compile(r"\w+")


def _simple_tokenize(text: str) -> List[str]:
    """
    中文：以「字元」為單位；英文：以單詞為單位。
    去除空白與標點。
    """
    if not text:
        return []

    # 判斷是否為（多數）中文
    chinese_ratio = sum("\u4e00" <= ch <= "\u9fff" for ch in text) / len(text)
    if chinese_ratio > 0.3:  # 粗略判斷
        return [ch for ch in text if ch.strip()]
    # 英文 / 其他語系以 word 為單位
    return _tok_split_en.findall(text.lower())


def _token_level_recall(gt: str, pred: str) -> float:
    gt_tokens = _simple_tokenize(gt)
    if not gt_tokens:
        return 0.0
    pred_tokens = _simple_tokenize(pred)

    # multiset 交集計算（保留重複）
    gt_counts = {}
    for tok in gt_tokens:
        gt_counts[tok] = gt_counts.get(tok, 0) + 1

    overlap = 0
    for tok in pred_tokens:
        if gt_counts.get(tok, 0) > 0:
            overlap += 1
            gt_counts[tok] -= 1

    return overlap / len(gt_tokens)


def calculate_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """計算 Token-level Recall 與 BERTScore F1"""
    if not reference or not hypothesis:
        return {
            "token_recall": 0.0,
            "bertscore_f1": 0.0,
            "response_length": len(hypothesis) if hypothesis else 0,
        }

    # Token-level Recall
    recall = _token_level_recall(reference, hypothesis)

    # BERTScore (中文)
    try:
        _, _, F1 = bertscore(
            cands=[hypothesis],
            refs=[reference],
            lang="zh",
            rescale_with_baseline=True,
        )
        bert_f1 = F1[0].item()
    except Exception as e:
        logger.warning(f"BERTScore 計算失敗: {e}")
        bert_f1 = 0.0

    return {
        "token_recall": recall,
        "bertscore_f1": bert_f1,
        "response_length": len(hypothesis),
    }
# -----------------------------

# ---------- RAG Engine 建立 ----------
def setup_engine(config: Dict[str, Any]) -> RAGEngine:
    model_manager = ModelManager(
        model_name=config["llm_model_name"],
        load_in_4bit=True,
    )
    db_manager = DBManager(
        embedding_model_name=config["embeddings_model_name"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        default_k_search=config["rag_top_k"],
    )

    if not db_manager.connect_db():
        logger.warning("資料庫連接失敗，將嘗試重建向量資料庫 …")
        db_manager.rebuild_db()

    return RAGEngine(
        model_manager=model_manager,
        db_manager=db_manager,
        system_prompt=config["system_prompt"],
        default_temperature=config["temperature"],
        default_max_new_tokens=config["max_new_tokens"],
    )
# -------------------------------------

# ---------- 資料載入 ----------
def load_qa_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    df = pd.read_csv(file_path)
    expected_cols = {
        "qa_id",
        "category",
        "sub_category",
        "type",
        "question",
        "ground_truth",
        "references",
    }
    if not expected_cols.issubset(df.columns):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"資料集缺少欄位: {', '.join(missing)}")

    logger.info(f"載入 {len(df)} 筆 QA，欄位: {df.columns.tolist()}")
    return df
# ----------------------------

# ---------- 主要評估函數 ----------
def evaluate_model(
    config_name: str,
    rag_engine: RAGEngine,
    dataset: pd.DataFrame,
    output_dir: str = "./results",
    category_filter: Optional[str] = None,
    limit: Optional[int] = None,
    rag_enabled: bool = True,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 類別篩選
    df_eval = (
        dataset[dataset["category"] == category_filter].copy()
        if category_filter
        else dataset.copy()
    )
    if limit and limit > 0:
        df_eval = df_eval.head(limit)

    os.makedirs(output_dir, exist_ok=True)

    # 插入欄位：model_answer、token_recall、bertscore_f1
    q_idx = df_eval.columns.get_loc("question")
    df_eval.insert(q_idx + 1, "model_answer", "")
    ref_idx = df_eval.columns.get_loc("references")
    df_eval.insert(ref_idx + 1, "token_recall", np.nan)
    df_eval.insert(ref_idx + 2, "bertscore_f1", np.nan)

    # 其他指標
    metric_columns_extra = ["response_length", "response_time"]
    for col in metric_columns_extra:
        df_eval[col] = np.nan
    df_eval["used_rag_resources"] = ""

    rag_status = "with_rag" if rag_enabled else "no_rag"
    result_file = f"{output_dir}/{config_name}_{rag_status}_{timestamp}.csv"

    logger.info(f"開始評估「{config_name}」，共 {len(df_eval)} 題")

    t_start_all = time.time()
    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="生成回答"):
        qa_id = row["qa_id"]
        question = row["question"]
        gt_answer = row["ground_truth"]

        if not isinstance(question, str) or not question.strip():
            logger.warning(f"問題 {qa_id} 內容為空，已跳過")
            continue

        try:
            t0 = time.time()
            answer = rag_engine.generate(
                query=question,
                use_rag=rag_enabled,
                enable_memory=False,
            )
            resp_time = time.time() - t0

            metrics = calculate_metrics(gt_answer, answer)
            metrics["response_time"] = resp_time

            rag_files = []
            if rag_enabled:
                for res in rag_engine.get_last_rag_resources():
                    rag_files.append(res.get("source_file_name", "未知"))

            # 寫回 DataFrame
            df_eval.at[idx, "model_answer"] = answer
            df_eval.at[idx, "token_recall"] = metrics["token_recall"]
            df_eval.at[idx, "bertscore_f1"] = metrics["bertscore_f1"]
            df_eval.at[idx, "response_length"] = metrics["response_length"]
            df_eval.at[idx, "response_time"] = metrics["response_time"]
            df_eval.at[idx, "used_rag_resources"] = ", ".join(rag_files)

            # 定期保存
            if idx % 5 == 0:
                df_eval.to_csv(result_file, index=False)

        except Exception as e:
            logger.error(f"處理問題 {qa_id} 發生錯誤: {e}", exc_info=True)
            df_eval.at[idx, "model_answer"] = f"處理錯誤: {e}"

    total_time = time.time() - t_start_all
    df_eval.to_csv(result_file, index=False)
    logger.info(f"評估完成，結果已保存至: {result_file}")

    # 總結指標
    avg_token_recall = df_eval["token_recall"].mean()
    avg_bertscore = df_eval["bertscore_f1"].mean()
    avg_resp_time = df_eval["response_time"].mean()
    logger.info("-" * 50)
    logger.info(
        f"模型摘要 - {config_name} ({'啟用RAG' if rag_enabled else '不使用RAG'}) | 題目數: {len(df_eval)}"
    )
    logger.info(f"平均 Token Recall : {avg_token_recall:.4f}")
    logger.info(f"平均 BERTScore F1  : {avg_bertscore:.4f}")
    logger.info(f"平均 Response Time: {avg_resp_time:.2f} 秒")
    logger.info(f"總耗時            : {total_time:.2f} 秒")
    logger.info("-" * 50)

    return result_file
# ----------------------------------------

# ---------- 配置載入 ----------
def load_model_config(cfg_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到配置檔: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfgs = json.load(f)
    logger.info(f"載入 {len(cfgs)} 組模型配置")
    return cfgs


def get_model_config_by_name(cfgs: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    return next((c for c in cfgs if c.get("config_name") == name), None)
# ----------------------------------------

# ---------- 主程式 ----------
def main():
    parser = argparse.ArgumentParser(description="ERAS QA 模型評估腳本")
    parser.add_argument("--config_name", required=True, help="模型配置名稱（在 model_config.json 中定義）")
    parser.add_argument("--config_path", default="./Eval/model_config.json", help="模型配置檔路徑")
    parser.add_argument("--dataset_path", default="./Eval/qa_dataset.csv", help="QA 資料集路徑")
    parser.add_argument("--output_dir", default="./Eval/results", help="結果輸出資料夾")
    parser.add_argument("--category", default=None, help="僅評估指定 category")
    parser.add_argument("--limit", type=int, default=None, help="限制題目數")
    parser.add_argument("--no_rag", action="store_true", help="停用 RAG")

    args = parser.parse_args()

    # 讀取配置
    configs = load_model_config(args.config_path)
    cfg = get_model_config_by_name(configs, args.config_name)
    if not cfg:
        avail = ", ".join(c.get("config_name", "未命名") for c in configs)
        logger.error(f"找不到配置「{args.config_name}」。可用配置: {avail}")
        return

    # 讀取資料集並建立 RAG Engine
    df = load_qa_dataset(args.dataset_path)
    rag_engine = setup_engine(cfg)

    # 評估
    evaluate_model(
        config_name=args.config_name,
        rag_engine=rag_engine,
        dataset=df,
        output_dir=args.output_dir,
        category_filter=args.category,
        limit=args.limit,
        rag_enabled=not args.no_rag,
    )


if __name__ == "__main__":
    main()