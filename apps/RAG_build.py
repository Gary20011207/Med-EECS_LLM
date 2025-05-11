# apps/RAG_build.py
import os
import shutil
import logging
from typing import List, Optional, Union # 引入 typing 中的類型提示

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document # 引入 Document 類型

# --- 全域組態設定 ---
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_PDF_FOLDER: str = "./PDFS"
DEFAULT_DB_PATH: str = "./VectorDB"

# --- 日誌設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

def reset_and_rebuild_vectordb(
    pdf_folder: str = DEFAULT_PDF_FOLDER,
    db_path: str = DEFAULT_DB_PATH,
    emb_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force_reset: bool = True,
    pdf_filenames: Optional[List[str]] = None 
) -> Optional[Chroma]:
    logger.info(f"開始向量資料庫處理：PDF來源='{pdf_folder}', DB路徑='{db_path}', 模型='{emb_model}'")
    if pdf_filenames:
        logger.info(f"指定處理檔案列表: {pdf_filenames}")

    # 1. 處理資料庫重置和目錄創建
    if force_reset and os.path.exists(db_path):
        logger.info(f"強制重置模式：正在刪除現有向量資料庫於 {db_path}")
        try:
            shutil.rmtree(db_path)
            logger.info(f"成功刪除資料庫目錄: {db_path}")
        except Exception as e:
            logger.error(f"刪除資料庫目錄 {db_path} 時出錯: {e}", exc_info=True)
            raise RuntimeError(f"無法刪除資料庫目錄 {db_path}，重建終止。") from e
    
    # 確保資料庫目錄存在，無論是否執行了 force_reset
    try:
        os.makedirs(db_path, exist_ok=True)
        logger.info(f"已確保資料庫目錄存在/已創建: {db_path}")
    except OSError as e:
        logger.error(f"創建資料庫目錄 {db_path} 失敗: {e}", exc_info=True)
        return None
            
    # 2. 準備文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, 
        add_start_index=True, 
    )
    
    # 3. 檢查並準備 PDF 資料夾
    if not os.path.exists(pdf_folder):
        logger.warning(f"PDF資料夾不存在: {pdf_folder}")
        try:
            os.makedirs(pdf_folder, exist_ok=True)
            logger.info(f"已成功創建PDF資料夾: {pdf_folder}")
        except OSError as e:
            logger.error(f"創建PDF資料夾 {pdf_folder} 失敗: {e}", exc_info=True)
            return None
            
    # 4. 確定要處理的 PDF 檔案列表
    files_to_process: List[str] = []
    if pdf_filenames: 
        for f_name in pdf_filenames:
            if not isinstance(f_name, str) or not f_name.lower().endswith(".pdf"):
                logger.warning(f"提供的檔案 '{f_name}' 不是有效的 PDF 檔名，將被忽略。")
                continue
            full_file_path = os.path.join(pdf_folder, f_name)
            if os.path.isfile(full_file_path): 
                files_to_process.append(f_name)
            else:
                logger.warning(f"指定的PDF檔案 '{f_name}' 在資料夾 '{pdf_folder}' 中未找到，將被忽略。")
    else: 
        try:
            all_folder_contents = os.listdir(pdf_folder)
            files_to_process = [f for f in all_folder_contents if f.lower().endswith(".pdf")]
            logger.info(f"未指定特定檔案，將掃描 '{pdf_folder}' 中的所有PDF。找到 {len(files_to_process)} 個PDF檔案。")
        except OSError as e:
            logger.error(f"讀取PDF資料夾 '{pdf_folder}' 內容失敗: {e}", exc_info=True)
            return None

    # 5. 如果沒有任何 PDF 檔案可處理
    if not files_to_process:
        logger.warning(f"在 '{pdf_folder}' 中最終沒有找到任何PDF檔案可供處理。")
        try:
            embedding_function = HuggingFaceEmbeddings(model_name=emb_model)
            vector_db = Chroma(
                embedding_function=embedding_function,
                persist_directory=db_path # 目錄已確保存在
            )
            vector_db.persist() 
            logger.info(f"已在 '{db_path}' 創建/加載了一個空的向量資料庫。")
            return vector_db
        except Exception as e:
            logger.error(f"創建嵌入模型或空的向量資料庫時出錯: {e}", exc_info=True)
            return None

    logger.info(f"準備處理 {len(files_to_process)} 個PDF檔案: {files_to_process}")
    
    # 6. 載入並分割選定的 PDF 文件
    all_document_chunks: List[Document] = []
    for i, pdf_file_name in enumerate(files_to_process, 1):
        file_path = os.path.join(pdf_folder, pdf_file_name)
        try:
            logger.info(f"正在處理檔案 ({i}/{len(files_to_process)}): {pdf_file_name}")
            loader = PyPDFLoader(file_path, extract_images=False) 
            pages = loader.load() 
            chunked_pages = text_splitter.split_documents(pages)
            for chunk in chunked_pages:
                chunk.metadata["source_pdf"] = pdf_file_name
            all_document_chunks.extend(chunked_pages)
            logger.debug(f"檔案 '{pdf_file_name}' 被成功處理並分割成 {len(chunked_pages)} 個文本塊。")
        except Exception as e:
            logger.error(f"處理PDF檔案 '{file_path}' 時發生錯誤: {e}", exc_info=True)
            
    # 7. 如果所有選定的 PDF 都處理失敗
    if not all_document_chunks:
        logger.warning("所有選定的PDF檔案均處理失敗或沒有可提取的內容。")
        try:
            embedding_function = HuggingFaceEmbeddings(model_name=emb_model)
            vector_db = Chroma(embedding_function=embedding_function, persist_directory=db_path)
            vector_db.persist()
            logger.info(f"已在 '{db_path}' 創建/加載空的向量資料庫 (因無成功處理的文本塊)。")
            return vector_db
        except Exception as e:
            logger.error(f"創建嵌入模型或空向量資料庫時出錯 (因無成功處理的文本塊): {e}", exc_info=True)
            return None

    logger.info(f"成功從選定的PDF檔案中處理了 {len(all_document_chunks)} 個文本塊。")
    
    # 8. 創建嵌入並建立/更新向量資料庫
    try:
        logger.info(f"正在使用嵌入模型 '{emb_model}' 創建文本嵌入...")
        embedding_function = HuggingFaceEmbeddings(model_name=emb_model)
    except Exception as e:
        logger.error(f"創建嵌入模型 '{emb_model}' 失敗: {e}", exc_info=True)
        return None

    try:
        logger.info(f"正在創建/更新向量資料庫並保存到 '{db_path}'...")
        vector_db = Chroma.from_documents(
            documents=all_document_chunks,
            embedding=embedding_function,
            persist_directory=db_path
        )
        vector_db.persist() 
        logger.info("向量資料庫處理完成!")
        return vector_db
    except Exception as e:
        logger.error(f"從文檔創建/更新向量資料庫並保存到 '{db_path}' 時失敗: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    logger.info("="*50)
    logger.info(f"開始獨立執行 RAG_build.py 腳本")
    logger.info(f"預設PDF資料夾: '{DEFAULT_PDF_FOLDER}', 預設DB路徑: '{DEFAULT_DB_PATH}'")
    logger.info("="*50)
    
    if not os.path.exists(DEFAULT_PDF_FOLDER) or not os.listdir(DEFAULT_PDF_FOLDER):
        logger.warning(f"預設PDF資料夾 '{DEFAULT_PDF_FOLDER}' 不存在或為空。")
        logger.warning("請先創建該資料夾並放入PDF文件，否則將創建一個空的向量資料庫。")

    logger.info("\n--- 測試案例 1: 處理指定的PDF檔案 ---")
    specific_files_to_test = []
    try:
        if os.path.exists(DEFAULT_PDF_FOLDER):
            all_pdfs_in_folder = [f for f in os.listdir(DEFAULT_PDF_FOLDER) if f.lower().endswith(".pdf")]
            if all_pdfs_in_folder:
                specific_files_to_test = all_pdfs_in_folder[:2] 
                logger.info(f"將嘗試使用特定檔案列表處理: {specific_files_to_test}")
            else:
                logger.info(f"'{DEFAULT_PDF_FOLDER}' 中沒有PDF檔案，無法執行特定檔案測試。")
        else:
            logger.info(f"'{DEFAULT_PDF_FOLDER}' 不存在，無法選取特定檔案進行測試。")
    except Exception as e:
        logger.error(f"準備特定檔案測試列表時出錯: {e}")

    if specific_files_to_test:
        try:
            test_db_path_specific = os.path.join(DEFAULT_DB_PATH, "test_specific_files_db")
            logger.info(f"將使用特定檔案列表重建資料庫: pdf_folder='{DEFAULT_PDF_FOLDER}', db_path='{test_db_path_specific}', force_reset=True")
            
            vectordb_instance_specific = reset_and_rebuild_vectordb(
                pdf_folder=DEFAULT_PDF_FOLDER,
                db_path=test_db_path_specific,
                force_reset=True,
                pdf_filenames=specific_files_to_test
            )
            
            if vectordb_instance_specific:
                count = vectordb_instance_specific._collection.count()
                logger.info(f"特定檔案資料庫操作成功。DB路徑: '{test_db_path_specific}', 記錄數: {count}")
                if count > 0:
                    results = vectordb_instance_specific.similarity_search("ERAS", k=1)
                    if results: logger.info(f"相似性搜索測試結果: {results[0].page_content[:100]}... (來源: {results[0].metadata.get('source_pdf')})")
                    else: logger.info("相似性搜索未返回結果。")
            else:
                logger.error(f"特定檔案資料庫操作失敗 (DB路徑: '{test_db_path_specific}')。")
        except Exception as e:
            logger.error(f"測試案例1執行時發生錯誤: {e}", exc_info=True)
    else:
        logger.info("跳過特定檔案列表測試。")

    logger.info("\n--- 測試案例 2: 處理所有PDF檔案 (使用預設DB路徑) ---")
    try:
        logger.info(f"將使用預設參數重建主資料庫: pdf_folder='{DEFAULT_PDF_FOLDER}', db_path='{DEFAULT_DB_PATH}', force_reset=True")
        
        vectordb_instance_all = reset_and_rebuild_vectordb(force_reset=True, pdf_filenames=None)
        
        if vectordb_instance_all:
            count = vectordb_instance_all._collection.count()
            logger.info(f"所有檔案資料庫操作成功。主DB ('{DEFAULT_DB_PATH}') 包含 {count} 條記錄。")
            if count > 0:
                results = vectordb_instance_all.similarity_search("術後恢復", k=1)
                if results: logger.info(f"相似性搜索測試結果: {results[0].page_content[:100]}... (來源: {results[0].metadata.get('source_pdf')})")
                else: logger.info("相似性搜索未返回結果。")
        else:
            logger.error("所有檔案資料庫操作失敗。")
    except Exception as e:
        logger.error(f"測試案例2執行時發生錯誤: {e}", exc_info=True)
    finally:
        logger.info("="*50)
        logger.info("RAG_build.py 腳本執行完畢。")
        logger.info("="*50)
