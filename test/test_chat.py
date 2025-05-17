# test_eras_core.py
import os
import logging
import time
from typing import List, Optional

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_eras_core')


# 1. 測試 ModelManager
def test_model_manager(model_name: str = "Qwen/Qwen2.5-14B-Instruct-1M", inactivity_timeout: int = 60):
    """測試 ModelManager 的基本功能"""
    from core.model_manager import ModelManager
    
    logger.info("===== 開始測試 ModelManager =====")
    
    # 初始化 ModelManager
    manager = ModelManager(
        model_name=model_name, 
        inactivity_timeout=inactivity_timeout,
        load_in_4bit=True  # 使用 4-bit 量化以節省記憶體
    )
    logger.info(f"ModelManager 初始化完成: {manager.model_name}")
    
    # 初始化模型
    logger.info("正在初始化模型...")
    model, tokenizer, max_context_length = manager.initialize()
    logger.info(f"模型初始化完成，上下文長度: {max_context_length}")
    
    # 獲取模型狀態
    status = manager.get_status()
    logger.info(f"模型狀態: {status}")
    
    # 測試 Token 計數
    test_text = "這是一個測試句子，用來測試 token 數量。"
    token_count = manager.count_tokens(test_text)
    logger.info(f"測試文本的 token 數量: {token_count}")
    
    # 測試生成回應
    logger.info("測試生成回應...")
    simple_prompt = "ERAS是什麼的縮寫？請用繁體中文回答。"
    response = manager.generate_response(
        prompt=simple_prompt,
        temperature=0.1,
        max_new_tokens=100
    )
    logger.info(f"模型直接回應: {response}")
    
    # 測試串流生成 (僅示例，實際測試可能需要自定義處理)
    logger.info("測試串流生成...")
    stream_prompt = "請簡單介紹 ERAS 的核心理念。"
    full_response = []
    try:
        for chunk in manager.generate_stream_response(
            prompt=stream_prompt,
            temperature=0.1,
            max_new_tokens=150
        ):
            full_response.append(chunk)
            # 實際應用中可能需要打印或其他處理
            print(chunk, end="", flush=True)
    except Exception as e:
        logger.error(f"串流生成錯誤: {e}")
    
    logger.info("\n串流生成完成，回應總長度: " + str(len("".join(full_response))))
    
    # 測試模型釋放資源
    logger.info("測試釋放模型資源...")
    manager.shutdown()
    logger.info("模型資源已釋放")
    
    logger.info("===== ModelManager 測試完成 =====\n")
    return manager

# 2. 測試 DBManager
def test_db_manager(rebuild: bool = False):
    """測試 DBManager 的基本功能"""
    from core.db_manager import DBManager
    
    logger.info("===== 開始測試 DBManager =====")
    
    # 初始化 DBManager
    db_manager = DBManager()
    logger.info(f"DBManager 初始化完成: {db_manager.embedding_model_name}")
    logger.info(f"使用集合名稱: {db_manager.collection_name}")
    
    # 連接資料庫
    logger.info("正在連接資料庫...")
    connected = db_manager.connect_db()
    logger.info(f"資料庫連接狀態: {'成功' if connected else '失敗'}")
    
    # 獲取資料庫狀態
    status = db_manager.get_status()
    logger.info(f"資料庫狀態: {status}")
    
    # 如果需要重建資料庫
    if rebuild:
        logger.info("正在重建資料庫...")
        rebuild_success = db_manager.rebuild_db()
        logger.info(f"資料庫重建: {'成功' if rebuild_success else '失敗'}")
        # 重建後再次獲取狀態
        status = db_manager.get_status()
        logger.info(f"重建後資料庫狀態: {status}")
    
    # 獲取可用源文件
    source_files = db_manager.get_available_source_files()
    logger.info(f"可用源文件: {source_files}")
    
    # 測試搜索
    logger.info("測試搜索功能...")
    query = "什麼是ERAS？請用繁體中文解釋。"
    results = db_manager.search(query)
    logger.info(f"搜索結果數量: {len(results)}")
    
    # 顯示搜索結果的前兩條（如果有）
    for i, doc in enumerate(results[:2], 1):
        metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
        logger.info(f"結果 {i}:")
        logger.info(f"內容 (前100字符): {doc.page_content[:100]}...")
        logger.info(f"元數據: {metadata_str}")
    
    logger.info("===== DBManager 測試完成 =====\n")
    return db_manager

# 3. 測試 RAGEngine
def test_rag_engine(model_manager, db_manager):
    """測試 RAGEngine 的基本功能"""
    from core.rag_engine import RAGEngine
    
    logger.info("===== 開始測試 RAGEngine =====")
    
    # 初始化 RAGEngine
    rag_engine = RAGEngine(model_manager, db_manager)
    logger.info("RAGEngine 初始化完成")
    
    # 測試獲取 RAG 信息
    query = "什麼是 ERAS 的護理重點？"
    logger.info(f"測試查詢: {query}")
    
    # 獲取 RAG 信息
    rag_info = rag_engine.get_rag_info(query)
    logger.info(f"RAG 信息結果數量: {rag_info['total_found']}")
    
    # 測試生成回應
    logger.info("測試 RAG 生成回應...")
    response = rag_engine.generate(
        query=query,
        temperature=0.1,
        max_new_tokens=200,
        use_rag=True
    )
    logger.info(f"RAG 生成回應: {response[:200]}...")
    
    # 獲取最後使用的 RAG 資源
    resources = rag_engine.get_last_rag_resources()
    logger.info(f"使用的 RAG 資源數量: {len(resources)}")
    
    # 測試禁用 RAG 的生成
    logger.info("測試禁用 RAG 的生成...")
    no_rag_response = rag_engine.generate(
        query=query,
        temperature=0.1,
        max_new_tokens=200,
        use_rag=False
    )
    logger.info(f"無 RAG 生成回應: {no_rag_response[:200]}...")
    
    # 測試串流生成 (實際應用可能需要更複雜的處理)
    logger.info("測試 RAG 串流生成...")
    stream_query = "列出ERAS術後飲食建議"
    full_response = []
    
    try:
        for chunk in rag_engine.stream(
            query=stream_query,
            temperature=0.1,
            max_new_tokens=300,
            use_rag=True
        ):
            full_response.append(chunk)
            # 實際應用中可能需要打印或其他處理
            print(chunk, end="", flush=True)
    except Exception as e:
        logger.error(f"RAG 串流生成錯誤: {e}")
    
    logger.info("\nRAG 串流生成完成，回應總長度: " + str(len("".join(full_response))))
    
    # 測試帶對話歷史的生成
    logger.info("測試帶對話歷史的 RAG 生成...")
    history = [
        {"role": "user", "content": "什麼是 ERAS？"},
        {"role": "assistant", "content": "ERAS 是 Enhanced Recovery After Surgery 的縮寫，指的是術後加速康復計劃。"}
    ]
    
    follow_up_query = "這種計劃有什麼好處？"
    response_with_history = rag_engine.generate(
        query=follow_up_query,
        temperature=0.1,
        max_new_tokens=200,
        use_rag=True,
        enable_memory=True,
        history=history
    )
    logger.info(f"帶歷史的 RAG 生成回應: {response_with_history[:200]}...")
    
    logger.info("===== RAGEngine 測試完成 =====\n")

# 主測試函數
def main():
    """執行所有測試"""
    logger.info("開始執行核心模組測試...")
    
    # 測試模型管理器
    model_manager = test_model_manager()
    
    # 測試資料庫管理器
    db_manager = test_db_manager(rebuild=False)  # 設為 True 以重建資料庫
    
    
    # 測試 RAG 引擎
    test_rag_engine(model_manager, db_manager)
    
    # 清理資源
    model_manager.shutdown()
    db_manager.disconnect()
    
    logger.info("所有測試完成！")

if __name__ == "__main__":
    main()