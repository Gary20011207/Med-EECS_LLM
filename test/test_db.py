# tests/test_db.py
import os
import sys
import time
import logging

# 添加專案根目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 先定義 logger
logger = logging.getLogger(__name__)

# 導入配置
try:
    from config import (
        EMBEDDINGS_MODEL_NAME,
        PDF_FOLDER,
        DB_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        RAG_TOP_K,
        LOG_FILE,
        LOG_LEVEL
    )
    logger.info("成功導入 config.py 中的配置")
    # 如果 config 中的 LOG_FILE 不是測試用的，則覆蓋
    if not LOG_FILE.startswith('test_'):
        LOG_FILE = "test_db.log"
except ImportError as e:
    logger.warning(f"無法導入 config.py: {e}，將使用預設配置")
    # 如果無法導入 config，設定預設值
    EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    PDF_FOLDER = "./PDFS"
    DB_PATH = "./VectorDB"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RAG_TOP_K = 5
    LOG_FILE = "test_db.log"
    LOG_LEVEL = "INFO"

# 導入模組
from core.db_manager import VectorDBManager

if __name__ == "__main__":
    # 設定日誌 (使用 config 中的設定)
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    # 設定檔案日誌和控制台日誌
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info(" VectorDBManager 專案資料夾測試")
    logger.info("="*60)
    logger.info(f"使用的配置:")
    logger.info(f"  嵌入模型: {EMBEDDINGS_MODEL_NAME}")
    logger.info(f"  PDF 資料夾: {PDF_FOLDER}")
    logger.info(f"  資料庫路徑: {DB_PATH}")
    logger.info(f"  文本塊大小: {CHUNK_SIZE}")
    logger.info(f"  文本塊重疊: {CHUNK_OVERLAP}")
    logger.info(f"  RAG Top-K: {RAG_TOP_K}")
    logger.info(f"  日誌檔案: {LOG_FILE}")
    logger.info(f"  日誌級別: {LOG_LEVEL}")
    
    # 創建 VectorDBManager 實例
    db_manager = VectorDBManager()
    
    try:
        logger.info("\n" + "="*50)
        logger.info("[步驟 1] 嘗試連接現有資料庫")
        logger.info("="*50)
        
        # 檢查資料庫狀態
        initial_status = db_manager.get_status()
        logger.info("初始資料庫狀態:")
        logger.info(f"  資料庫存在: {initial_status['db_exists']}")
        logger.info(f"  資料庫路徑: {initial_status['db_path']}")
        logger.info(f"  PDF 資料夾: {initial_status['pdf_folder']}")
        logger.info(f"  PDF 檔案數: {initial_status['pdf_count']}")
        
        # 嘗試連接資料庫
        logger.info("\n嘗試連接資料庫...")
        connected_db = db_manager.connect_db()
        
        if connected_db:
            logger.info("✓ 成功連接到現有資料庫")
            status = db_manager.get_status()
            logger.info(f"  資料庫記錄數: {status['record_count']}")
            logger.info(f"  資料庫中的源文件數: {status['db_source_files_count']}")
            logger.info(f"  源文件列表: {status['db_source_files']}")
            
            # 判斷是否需要更新
            pdf_files = status['pdf_files']
            db_source_files = status['db_source_files']
            
            # 簡單判斷：如果 PDF 檔案數量與資料庫中的不同，則認為需要更新
            need_update = len(pdf_files) != len(db_source_files)
            if not need_update:
                # 檢查檔案名是否一致
                need_update = set(pdf_files) != set(db_source_files)
            
            if need_update:
                logger.info("⚠ 檢測到 PDF 檔案有變化，需要重建資料庫")
            else:
                logger.info("✓ 資料庫已是最新，無需更新")
        else:
            logger.info("⚠ 無法連接到現有資料庫，需要建立新資料庫")
        
        logger.info("\n" + "="*50)
        logger.info("[步驟 2] 強制清除重新建立資料庫")
        logger.info("="*50)
        
        logger.info("執行強制重建資料庫...")
        # 重建資料庫（會自動處理 PDF 資料夾不存在的情況）
        rebuilt_db = db_manager.reset_and_rebuild_db(force_reset=True)
        
        if rebuilt_db:
            logger.info("✓ 資料庫重建成功")
            # 檢查重建狀態
            rebuild_status = db_manager.get_rebuild_status()
            logger.info(f"重建狀態: {rebuild_status['status']}, 進度: {rebuild_status['progress']}%")
            logger.info(f"狀態訊息: {rebuild_status['message']}")
            
            # 檢查最終狀態
            final_status = db_manager.get_status()
            logger.info("重建後的資料庫狀態:")
            logger.info(f"  記錄數: {final_status['record_count']}")
            logger.info(f"  處理檔案數: {final_status['pdf_count']}")
            logger.info(f"  資料庫中的源文件數: {final_status['db_source_files_count']}")
        else:
            logger.error("✗ 資料庫重建失敗")
        
        logger.info("\n" + "="*50)
        logger.info("[步驟 3] 重新連接資料庫")
        logger.info("="*50)
        
        # 重新連接以確保狀態同步
        logger.info("重新連接資料庫...")
        final_db = db_manager.connect_db()
        
        if final_db:
            logger.info("✓ 成功重新連接到資料庫")
            status = db_manager.get_status()
            logger.info(f"  資料庫記錄數: {status['record_count']}")
            logger.info(f"  預設 Top-K: {status['default_top_k']}")
        else:
            logger.error("✗ 重新連接資料庫失敗")
        
        logger.info("\n" + "="*50)
        logger.info("[步驟 4] 檢索測試")
        logger.info("="*50)
        
        if final_db and status['record_count'] > 0:
            # 測試 1: 全文檢索
            logger.info("\n[子測試 4.1] 全文檢索測試...")
            all_files_results = db_manager.search("手術", k=5)
            if all_files_results:
                logger.info(f"✓ 全文檢索成功，找到 {len(all_files_results)} 個結果")
                for i, result in enumerate(all_files_results):
                    source_file = result.metadata.get('source_file_name', '未知')
                    chunk_index = result.metadata.get('chunk_index', 0)
                    logger.info(f"  結果 {i+1} (來源: {source_file}, 塊: {chunk_index}): {result.page_content[:80]}...")
            else:
                logger.info("全文檢索未返回結果")
            
            # 測試 2: 指定文件檢索
            logger.info("\n[子測試 4.2] 指定文件檢索測試...")
            db_source_files = status['db_source_files']
            
            if db_source_files:
                # 選取前三個文件進行測試
                test_files = db_source_files[:3]
                logger.info(f"測試文件: {test_files}")
                
                for test_file in test_files:
                    logger.info(f"\n  測試文件: {test_file}")
                    specific_results = db_manager.search("患者", k=2, source_files=[test_file])
                    if specific_results:
                        logger.info(f"  ✓ 在 {test_file} 中找到 {len(specific_results)} 個結果")
                        for j, result in enumerate(specific_results):
                            chunk_index = result.metadata.get('chunk_index', 0)
                            logger.info(f"    結果 {j+1} (塊: {chunk_index}): {result.page_content[:60]}...")
                    else:
                        logger.info(f"  在 {test_file} 中未找到相關結果")
            else:
                logger.warning("  沒有源文件可以進行指定文件檢索測試")
            
            # 測試 3: 使用預設 top-k 檢索
            logger.info("\n[子測試 4.3] 預設 Top-K 檢索測試...")
            default_results = db_manager.search("恢復")  # 不指定 k，使用預設值
            logger.info(f"使用預設 Top-K={status['default_top_k']}，找到 {len(default_results)} 個結果")
            if default_results:
                logger.info("前兩個結果:")
                for i, result in enumerate(default_results[:2]):
                    source_file = result.metadata.get('source_file_name', '未知')
                    logger.info(f"  結果 {i+1} (來源: {source_file}): {result.page_content[:70]}...")
        else:
            logger.warning("⚠ 資料庫為空或未連接，跳過檢索測試")
    
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}", exc_info=True)
    
    finally:
        logger.info("\n" + "="*60)
        logger.info(" VectorDBManager 專案資料夾測試完畢")
        logger.info("="*60)