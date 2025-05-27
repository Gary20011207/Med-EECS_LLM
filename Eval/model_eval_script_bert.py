#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERAS QA 模型評估腳本
用於評估不同模型配置對 ERAS 問答數據集的表現
"""

import os
import time
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
from tqdm import tqdm

# 導入核心模組
from core.model_manager import ModelManager
from core.db_manager import DBManager
from core.rag_engine import RAGEngine

# 評估指標：BERTScore（支援中文）
from bert_score import score as bertscore

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"model_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_eval")

def calculate_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """使用 BERTScore（支援中文）計算模型回答品質"""
    if not hypothesis or not reference:
        return {
            "bertscore_f1": 0.0,
            "response_length": len(hypothesis) if hypothesis else 0
        }

    try:
        P, R, F1 = bertscore(
            cands=[hypothesis],
            refs=[reference],
            lang="zh",
            rescale_with_baseline=True
        )
        return {
            "bertscore_f1": F1[0].item(),
            "response_length": len(hypothesis)
        }
    except Exception as e:
        logger.warning(f"BERTScore 計算失敗: {e}")
        return {
            "bertscore_f1": 0.0,
            "response_length": len(hypothesis)
        }

def setup_engine(config: Dict[str, Any]) -> RAGEngine:
    """設置RAG引擎"""
    model_manager = ModelManager(
        model_name=config["llm_model_name"],
        load_in_4bit=True
    )

    db_manager = DBManager(
        embedding_model_name=config["embeddings_model_name"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        default_k_search=config["rag_top_k"]
    )

    connected = db_manager.connect_db()
    if not connected:
        logger.warning("資料庫連接失敗，將嘗試從PDFS資料夾重建")
        db_manager.rebuild_db()

    rag_engine = RAGEngine(
        model_manager=model_manager,
        db_manager=db_manager,
        system_prompt=config["system_prompt"],
        default_temperature=config["temperature"],
        default_max_new_tokens=config["max_new_tokens"]
    )

    return rag_engine

def load_qa_dataset(file_path: str) -> pd.DataFrame:
    """載入QA數據集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"載入{len(df)}個QA項目，列: {df.columns.tolist()}")
    return df

def evaluate_model(config_name: str, rag_engine: RAGEngine, dataset: pd.DataFrame, 
                   output_dir: str = "./results", category_filter: Optional[str] = None,
                   limit: Optional[int] = None, rag_enabled: bool = True) -> str:
    """評估模型並返回結果檔案路徑"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if category_filter:
        filtered_data = dataset[dataset["category"] == category_filter].copy()
        if len(filtered_data) == 0:
            logger.warning(f"類別 '{category_filter}' 沒有找到任何數據，將使用全部數據")
            filtered_data = dataset.copy()
    else:
        filtered_data = dataset.copy()

    if limit and limit > 0 and limit < len(filtered_data):
        filtered_data = filtered_data.head(limit)

    os.makedirs(output_dir, exist_ok=True)

    rag_status = "with_rag" if rag_enabled else "no_rag"
    result_file = f"{output_dir}/{config_name}_{rag_status}_{timestamp}.csv"

    result_df = filtered_data.copy()
    result_df["llm_answer"] = ""
    result_df["human_eval"] = ""  # 留空，供人工評估

    # 評估指標欄位
    metric_columns = ["bertscore_f1", "response_length", "response_time"]
    for col in metric_columns:
        result_df[col] = np.nan

    result_df["used_rag_resources"] = ""

    start_time_total = time.time()
    logger.info(f"開始評估 '{config_name}'，共{len(filtered_data)}個問題")

    for idx, row in tqdm(filtered_data.iterrows(), total=len(filtered_data), desc="生成回答"):
        qa_id = row["qa_id"]
        question = row["question"]
        reference = row["reference_answer"]

        logger.info(f"正在處理問題 {qa_id}: {question[:50]}...")

        if not isinstance(question, str) or not question.strip():
            logger.warning(f"問題ID {qa_id} 的問題內容為空，跳過")
            continue

        try:
            start_time = time.time()

            answer = rag_engine.generate(
                query=question,
                use_rag=rag_enabled,
                enable_memory=False
            )

            response_time = time.time() - start_time

            metrics = calculate_metrics(reference, answer)
            metrics["response_time"] = response_time

            rag_resources = []
            if rag_enabled:
                for res in rag_engine.get_last_rag_resources():
                    rag_resources.append(res.get("source_file_name", "未知"))

            result_df.at[idx, "llm_answer"] = answer
            for metric_name, metric_value in metrics.items():
                result_df.at[idx, metric_name] = metric_value
            result_df.at[idx, "used_rag_resources"] = ", ".join(rag_resources)

            if idx % 5 == 0:
                result_df.to_csv(result_file, index=False)

        except Exception as e:
            logger.error(f"處理問題 {qa_id} 時出錯: {e}", exc_info=True)
            result_df.at[idx, "llm_answer"] = f"處理錯誤: {str(e)}"

    total_time = time.time() - start_time_total
    logger.info(f"評估完成，總時間: {total_time:.2f}秒，每個問題平均: {total_time/len(filtered_data):.2f}秒")

    result_df.to_csv(result_file, index=False)
    logger.info(f"評估結果已保存至: {result_file}")

    average_metrics = {col: result_df[col].mean() for col in metric_columns if col != "response_time"}
    average_metrics["response_time"] = result_df["response_time"].mean()

    logger.info("-" * 50)
    logger.info(f"模型評估摘要 - {config_name} ({'啟用RAG' if rag_enabled else '不使用RAG'})")
    logger.info(f"測試類別: {category_filter if category_filter else '全部'}")
    logger.info(f"問題數量: {len(filtered_data)}")
    logger.info(f"總時間: {total_time:.2f}秒")
    logger.info("平均指標:")
    for metric, value in average_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("-" * 50)

    return result_file

def load_model_config(config_path: str) -> List[Dict[str, Any]]:
    """載入模型配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)

    logger.info(f"已載入{len(configs)}個模型配置")
    return configs

def get_model_config_by_name(configs: List[Dict[str, Any]], config_name: str) -> Optional[Dict[str, Any]]:
    """根據名稱獲取模型配置"""
    for config in configs:
        if config.get("config_name") == config_name:
            return config
    return None

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="ERAS QA 模型評估腳本")
    parser.add_argument("--config_name", type=str, required=True,
                        help="要評估的模型配置名稱（需在model_config.json中定義）")
    parser.add_argument("--config_path", type=str, default="./Eval/model_config.json",
                        help="模型配置文件路徑")
    parser.add_argument("--dataset_path", type=str, default="./Eval/qa_dataset.csv",
                        help="問答數據集路徑")
    parser.add_argument("--output_dir", type=str, default="./Eval/results",
                        help="結果輸出目錄")
    parser.add_argument("--category", type=str, default=None,
                        help="篩選特定類別的問題")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制評估的問題數量")
    parser.add_argument("--no_rag", action="store_true",
                        help="不使用RAG功能進行測試")

    args = parser.parse_args()

    model_configs = load_model_config(args.config_path)
    selected_config = get_model_config_by_name(model_configs, args.config_name)

    if not selected_config:
        logger.error(f"找不到名為 '{args.config_name}' 的配置，請檢查 {args.config_path}")
        available_configs = [cfg.get("config_name", "未命名") for cfg in model_configs]
        logger.info(f"可用的配置有: {', '.join(available_configs)}")
        return

    dataset = load_qa_dataset(args.dataset_path)
    rag_engine = setup_engine(selected_config)

    rag_enabled = not args.no_rag
    result_file = evaluate_model(
        config_name=args.config_name,
        rag_engine=rag_engine,
        dataset=dataset,
        output_dir=args.output_dir,
        category_filter=args.category,
        limit=args.limit,
        rag_enabled=rag_enabled
    )

    logger.info(f"評估完成，結果已保存至: {result_file}")

if __name__ == "__main__":
    main()